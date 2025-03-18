from __future__ import annotations

import asyncio
import datetime
import inspect
import logging
import sys
import time
from typing import Awaitable, Collection, Optional

from pixeltable import env, func
from pixeltable.config import Config

from .globals import Dispatcher, FnCallArgs, Scheduler

_logger = logging.getLogger('pixeltable')


class RateLimitsScheduler(Scheduler):
    """
    Scheduler for FunctionCalls with a RateLimitsInfo pool, which provides information about actual resource usage.

    Scheduling strategy:
    - try to stay below resource limits by utilizing reported RateLimitInfo.remaining
    - also take into account the estimated resource usage for in-flight requests
      (obtained via RateLimitsInfo.get_request_resources())
    - issue synchronous requests when we don't have a RateLimitsInfo yet or when we depleted a resource and need to
      wait for a reset

    TODO:
    - limit the number of in-flight requests based on the open file limit
    """

    get_request_resources_param_names: list[str]  # names of parameters of RateLimitsInfo.get_request_resources()

    # scheduling-related state
    pool_info: Optional[env.RateLimitsInfo]
    est_usage: dict[str, int]  # value per resource; accumulated estimates since the last util. report

    num_in_flight: int  # unfinished tasks
    request_completed: asyncio.Event

    total_requests: int
    total_retried: int

    TIME_FORMAT = '%H:%M.%S %f'
    MAX_RETRIES = 10

    def __init__(self, resource_pool: str, dispatcher: Dispatcher):
        super().__init__(resource_pool, dispatcher)
        loop_task = asyncio.create_task(self._main_loop())
        self.dispatcher.register_task(loop_task)
        self.pool_info = None  # initialized in _main_loop by the first request
        self.est_usage = {}
        self.num_in_flight = 0
        self.request_completed = asyncio.Event()
        self.total_requests = 0
        self.total_retried = 0
        self.get_request_resources_param_names = []

    @classmethod
    def matches(cls, resource_pool: str) -> bool:
        return resource_pool.startswith('rate-limits:')

    def submit(self, item: FnCallArgs) -> None:
        self.queue.put_nowait(self.QueueItem(item, 0))

    def _set_pool_info(self) -> None:
        """Initialize pool_info with the RateLimitsInfo for the resource pool, if available"""
        if self.pool_info is not None:
            return
        self.pool_info = env.Env.get().get_resource_pool_info(self.resource_pool, None)
        if self.pool_info is None:
            return
        assert isinstance(self.pool_info, env.RateLimitsInfo)
        assert hasattr(self.pool_info, 'get_request_resources')
        sig = inspect.signature(self.pool_info.get_request_resources)
        self.get_request_resources_param_names = [p.name for p in sig.parameters.values()]
        self.est_usage = {r: 0 for r in self._resources}

    async def _main_loop(self) -> None:
        item: Optional[RateLimitsScheduler.QueueItem] = None
        while True:
            if item is None:
                item = await self.queue.get()
                if item.num_retries > 0:
                    self.total_retried += 1

            now = datetime.datetime.now(tz=datetime.timezone.utc)
            if self.pool_info is None or not self.pool_info.is_initialized():
                # wait for a single request to get rate limits
                _logger.debug(f'initializing rate limits for {self.resource_pool}')
                await self._exec(item.request, item.num_retries, is_task=False)
                _logger.debug(f'initialized rate limits for {self.resource_pool}')
                item = None
                # if this was the first request, it created the pool_info
                if self.pool_info is None:
                    self._set_pool_info()
                continue

            # check rate limits
            _logger.debug(f'checking rate limits for {self.resource_pool}')
            request_resources = self._get_request_resources(item.request)
            limits_info = self._check_resource_limits(request_resources)
            aws: list[Awaitable[None]] = []
            completed_aw: Optional[asyncio.Task] = None
            wait_for_reset: Optional[asyncio.Task] = None
            if limits_info is not None:
                # limits_info's resource is depleted, wait for capacity to free up

                if self.num_in_flight > 0:
                    # a completed request can free up capacity
                    self.request_completed.clear()
                    completed_aw = asyncio.create_task(self.request_completed.wait())
                    aws.append(completed_aw)
                    _logger.debug(f'waiting for completed request for {self.resource_pool}')

                reset_at = limits_info.reset_at
                if reset_at > now:
                    # we're waiting for the rate limit to reset
                    wait_for_reset = asyncio.create_task(asyncio.sleep((reset_at - now).total_seconds()))
                    aws.append(wait_for_reset)
                    _logger.debug(f'waiting for rate limit reset for {self.resource_pool}')

            if len(aws) > 0:
                # we have something to wait for
                done, pending = await asyncio.wait(aws, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                if completed_aw in done:
                    _logger.debug(f'wait(): completed request for {self.resource_pool}')
                if wait_for_reset in done:
                    _logger.debug(f'wait(): rate limit reset for {self.resource_pool}')
                    # force waiting for another rate limit report before making any scheduling decisions
                    self.pool_info.reset()
                # re-evaluate current capacity for current item
                continue

            # we have a new in-flight request
            for resource, val in request_resources.items():
                self.est_usage[resource] += val
            _logger.debug(f'creating task for {self.resource_pool}')
            self.num_in_flight += 1
            task = asyncio.create_task(self._exec(item.request, item.num_retries, is_task=True))
            self.dispatcher.register_task(task)
            item = None

    @property
    def _resources(self) -> Collection[str]:
        return self.pool_info.resource_limits.keys() if self.pool_info is not None else []

    def _get_request_resources(self, request: FnCallArgs) -> dict[str, int]:
        kwargs_batch = request.fn_call.get_param_values(self.get_request_resources_param_names, request.rows)
        if not request.is_batched:
            return self.pool_info.get_request_resources(**kwargs_batch[0])
        else:
            batch_kwargs = {k: [d[k] for d in kwargs_batch] for k in kwargs_batch[0]}
            constant_kwargs, batch_kwargs = request.pxt_fn.create_batch_kwargs(batch_kwargs)
            return self.pool_info.get_request_resources(**constant_kwargs, **batch_kwargs)

    def _check_resource_limits(self, request_resources: dict[str, int]) -> Optional[env.RateLimitInfo]:
        """Returns the most depleted resource, relative to its limit, or None if all resources are within limits"""
        candidates: list[tuple[env.RateLimitInfo, float]] = []  # (info, relative usage)
        for resource, usage in request_resources.items():
            # 0.05: leave some headroom, we don't have perfect information
            info = self.pool_info.resource_limits[resource]
            est_remaining = info.remaining - self.est_usage[resource] - usage
            if est_remaining < 0.05 * info.limit:
                candidates.append((info, est_remaining / info.limit))
        if len(candidates) == 0:
            return None
        return min(candidates, key=lambda x: x[1])[0]

    async def _exec(self, request: FnCallArgs, num_retries: int, is_task: bool) -> None:
        assert all(not row.has_val[request.fn_call.slot_idx] for row in request.rows)
        assert all(not row.has_exc(request.fn_call.slot_idx) for row in request.rows)

        try:
            start_ts = datetime.datetime.now(tz=datetime.timezone.utc)
            pxt_fn = request.fn_call.fn
            assert isinstance(pxt_fn, func.CallableFunction)
            _logger.debug(
                f'scheduler {self.resource_pool}: start evaluating slot {request.fn_call.slot_idx}, batch_size={len(request.rows)}'
            )
            self.total_requests += 1
            if request.is_batched:
                batch_result = await pxt_fn.aexec_batch(*request.batch_args, **request.batch_kwargs)
                assert len(batch_result) == len(request.rows)
                for row, result in zip(request.rows, batch_result):
                    row[request.fn_call.slot_idx] = result
            else:
                result = await pxt_fn.aexec(*request.args, **request.kwargs)
                request.row[request.fn_call.slot_idx] = result
            end_ts = datetime.datetime.now(tz=datetime.timezone.utc)
            _logger.debug(
                f'scheduler {self.resource_pool}: evaluated slot {request.fn_call.slot_idx} in {end_ts - start_ts}, batch_size={len(request.rows)}'
            )

            # purge accumulated usage estimate, now that we have a new report
            self.est_usage = {r: 0 for r in self._resources}

            self.dispatcher.dispatch(request.rows)
        except Exception as exc:
            _logger.debug(f'scheduler {self.resource_pool}: exception in slot {request.fn_call.slot_idx}: {exc}')
            if self.pool_info is None:
                # our pool info should be available at this point
                self._set_pool_info()
            assert self.pool_info is not None
            if num_retries < self.MAX_RETRIES:
                retry_delay = self.pool_info.get_retry_delay(exc)
                if retry_delay is not None:
                    self.total_retried += 1
                    _logger.debug(f'scheduler {self.resource_pool}: retrying in {retry_delay} seconds')
                    await asyncio.sleep(retry_delay)
                    self.queue.put_nowait(self.QueueItem(request, num_retries + 1))
                    return
            # TODO: update resource limits reported in exc.response.headers, if present

            # record the exception
            _, _, exc_tb = sys.exc_info()
            for row in request.rows:
                row.set_exc(request.fn_call.slot_idx, exc)
            self.dispatcher.dispatch_exc(request.rows, request.fn_call.slot_idx, exc_tb)
        finally:
            _logger.debug(f'Scheduler stats: #requests={self.total_requests}, #retried={self.total_retried}')
            if is_task:
                self.num_in_flight -= 1
                self.request_completed.set()


class RequestRateScheduler(Scheduler):
    """
    Scheduler for FunctionCalls with a fixed request rate limit and no runtime resource usage reports.

    Rate limits are supplied in the config, in one of two ways:
    - resource_pool='request-rate:<endpoint>':
      * a single rate limit for all calls against that endpoint
      * in the config: section '<endpoint>', key 'rate_limit'
    - resource_pool='request-rate:<endpoint>:<model>':
        * a single rate limit for all calls against that model
        * in the config: section '<endpoint>.rate_limits', key '<model>'
    - if no rate limit is found in the config, uses a default of 600 RPM

    TODO:
    - adaptive rate limiting based on 429 errors
    """

    secs_per_request: float  # inverted rate limit
    num_in_flight: int
    total_requests: int
    total_retried: int

    TIME_FORMAT = '%H:%M.%S %f'
    MAX_RETRIES = 10
    DEFAULT_RATE_LIMIT = 600  # requests per minute

    def __init__(self, resource_pool: str, dispatcher: Dispatcher):
        super().__init__(resource_pool, dispatcher)
        loop_task = asyncio.create_task(self._main_loop())
        self.dispatcher.register_task(loop_task)
        self.num_in_flight = 0
        self.total_requests = 0
        self.total_retried = 0

        # try to get the rate limit from the config
        elems = resource_pool.split(':')
        section: str
        key: str
        if len(elems) == 2:
            # resource_pool: request-rate:endpoint
            _, endpoint = elems
            section = endpoint
            key = 'rate_limit'
        else:
            # resource_pool: request-rate:endpoint:model
            assert len(elems) == 3
            _, endpoint, model = elems
            section = f'{endpoint}.rate_limits'
            key = model
        requests_per_min = Config.get().get_int_value(key, section=section)
        requests_per_min = requests_per_min or self.DEFAULT_RATE_LIMIT
        self.secs_per_request = 1 / (requests_per_min / 60)

    @classmethod
    def matches(cls, resource_pool: str) -> bool:
        return resource_pool.startswith('request-rate:')

    async def _main_loop(self) -> None:
        last_request_ts = 0.0
        while True:
            item = await self.queue.get()
            if item.num_retries > 0:
                self.total_retried += 1
            now = time.monotonic()
            if now - last_request_ts < self.secs_per_request:
                wait_duration = self.secs_per_request - (now - last_request_ts)
                _logger.debug(f'waiting for {wait_duration} for {self.resource_pool}')
                await asyncio.sleep(wait_duration)

            last_request_ts = time.monotonic()
            if item.num_retries > 0:
                # the last request encountered some problem: retry it synchronously, to wait for the problem to pass
                _logger.debug(f'retrying request for {self.resource_pool}: #retries={item.num_retries}')
                await self._exec(item.request, item.num_retries, is_task=False)
                _logger.debug(f'retried request for {self.resource_pool}: #retries={item.num_retries}')
            else:
                _logger.debug(f'creating task for {self.resource_pool}')
                self.num_in_flight += 1
                task = asyncio.create_task(self._exec(item.request, item.num_retries, is_task=True))
                self.dispatcher.register_task(task)

    async def _exec(self, request: FnCallArgs, num_retries: int, is_task: bool) -> None:
        assert all(not row.has_val[request.fn_call.slot_idx] for row in request.rows)
        assert all(not row.has_exc(request.fn_call.slot_idx) for row in request.rows)

        try:
            start_ts = datetime.datetime.now(tz=datetime.timezone.utc)
            pxt_fn = request.fn_call.fn
            assert isinstance(pxt_fn, func.CallableFunction)
            _logger.debug(
                f'scheduler {self.resource_pool}: start evaluating slot {request.fn_call.slot_idx}, batch_size={len(request.rows)}'
            )
            self.total_requests += 1
            if request.is_batched:
                batch_result = await pxt_fn.aexec_batch(*request.batch_args, **request.batch_kwargs)
                assert len(batch_result) == len(request.rows)
                for row, result in zip(request.rows, batch_result):
                    row[request.fn_call.slot_idx] = result
            else:
                result = await pxt_fn.aexec(*request.args, **request.kwargs)
                request.row[request.fn_call.slot_idx] = result
            end_ts = datetime.datetime.now(tz=datetime.timezone.utc)
            _logger.debug(
                f'scheduler {self.resource_pool}: evaluated slot {request.fn_call.slot_idx} in {end_ts - start_ts}, batch_size={len(request.rows)}'
            )
            self.dispatcher.dispatch(request.rows)

        except Exception as exc:
            # TODO: which exception can be retried?
            _logger.debug(f'exception for {self.resource_pool}: {exc}')
            status = getattr(exc, 'status', None)
            _logger.debug(f'type={type(exc)} has_status={hasattr(exc, "status")} status={status}')
            if num_retries < self.MAX_RETRIES:
                self.queue.put_nowait(self.QueueItem(request, num_retries + 1))
                return

            # record the exception
            _, _, exc_tb = sys.exc_info()
            for row in request.rows:
                row.set_exc(request.fn_call.slot_idx, exc)
            self.dispatcher.dispatch_exc(request.rows, request.fn_call.slot_idx, exc_tb)
        finally:
            _logger.debug(
                f'Scheduler stats: #in-flight={self.num_in_flight} #requests={self.total_requests}, #retried={self.total_retried}'
            )
            if is_task:
                self.num_in_flight -= 1


# all concrete Scheduler subclasses that implement matches()
SCHEDULERS = [RateLimitsScheduler, RequestRateScheduler]
