from typing import List, Optional, Tuple
from dataclasses import dataclass, field
import sys
import logging
import time

import numpy as np
import nos
from tqdm.autonotebook import tqdm

from .data_row_batch import DataRowBatch
from .exec_node import ExecNode
import pixeltable.exprs as exprs
import pixeltable.func as func
import pixeltable.env as env


_logger = logging.getLogger('pixeltable')

class ExprEvalNode(ExecNode):
    """Materializes expressions
    """
    @dataclass
    class Cohort:
        """List of exprs that form an evaluation context and contain calls to at most one NOS function"""
        exprs: List[exprs.Expr]
        nos_function: Optional[func.NOSFunction]
        segment_ctxs: List[exprs.RowBuilder.EvalCtx]
        target_slot_idxs: List[int]
        batch_size: int = 8

    def __init__(
            self, row_builder: exprs.RowBuilder, output_exprs: List[exprs.Expr], input_exprs: List[exprs.Expr],
            ignore_errors: bool, input: ExecNode
    ):
        super().__init__(row_builder, output_exprs, input_exprs, input)
        self.input_exprs = input_exprs
        input_slot_idxs = {e.slot_idx for e in input_exprs}
        # we're only materializing exprs that are not already in the input
        self.target_exprs = [e for e in output_exprs if e.slot_idx not in input_slot_idxs]
        self.ignore_errors = ignore_errors  # if False, raise exc.ExprEvalError on error in _exec_cohort()
        self.pbar: Optional[tqdm] = None
        self.cohorts: List[List[ExprEvalNode.Cohort]] = []
        self._create_cohorts()

    def __next__(self) -> DataRowBatch:
        input_batch = next(self.input)
        # compute target exprs
        for cohort in self.cohorts:
            self._exec_cohort(cohort, input_batch)
        _logger.debug(f'ExprEvalNode: returning {len(input_batch)} rows')
        return input_batch

    def _open(self) -> None:
        if self.ctx.show_pbar:
            self.pbar = tqdm(total=len(self.target_exprs) * self.ctx.num_rows, desc='Computing cells', unit='cells')

    def _close(self) -> None:
        if self.pbar is not None:
            self.pbar.close()

    def _get_nos_fn(self, expr: exprs.Expr) -> Optional[func.NOSFunction]:
        """Get NOSFunction if expr is a call to a NOS function, else None."""
        if not isinstance(expr, exprs.FunctionCall):
            return None
        return expr.fn if isinstance(expr.fn, func.NOSFunction) else None

    def _is_nos_call(self, expr: exprs.Expr) -> bool:
        return self._get_nos_fn(expr) is not None

    def _create_cohorts(self) -> None:
        all_exprs = self.row_builder.get_dependencies(self.target_exprs)
        # break up all_exprs into cohorts such that each cohort contains calls to at most one NOS function;
        # seed the cohorts with only the nos calls
        cohorts: List[List[exprs.Expr]] = []
        current_nos_function: Optional[func.NOSFunction] = None
        for e in all_exprs:
            if not self._is_nos_call(e):
                continue
            if current_nos_function is None or current_nos_function != e.fn:
                # create a new cohort
                cohorts.append([])
                current_nos_function = e.fn
            cohorts[-1].append(e)

        # expand the cohorts to include all exprs that are in the same evaluation context as the NOS calls;
        # cohorts are evaluated in order, so we can exclude the target slots from preceding cohorts and input slots
        exclude = set([e.slot_idx for e in self.input_exprs])
        all_target_slot_idxs = set([e.slot_idx for e in self.target_exprs])
        target_slot_idxs: List[List[int]] = []  # the ones materialized by each cohort
        for i in range(len(cohorts)):
            cohorts[i] = self.row_builder.get_dependencies(
                cohorts[i], exclude=[self.row_builder.unique_exprs[slot_idx] for slot_idx in exclude])
            target_slot_idxs.append(
                [e.slot_idx for e in cohorts[i] if e.slot_idx in all_target_slot_idxs])
            exclude.update(target_slot_idxs[-1])

        all_cohort_slot_idxs = set([e.slot_idx for cohort in cohorts for e in cohort])
        remaining_slot_idxs = set(all_target_slot_idxs) - all_cohort_slot_idxs
        if len(remaining_slot_idxs) > 0:
            cohorts.append(self.row_builder.get_dependencies(
                [self.row_builder.unique_exprs[slot_idx] for slot_idx in remaining_slot_idxs],
                exclude=[self.row_builder.unique_exprs[slot_idx] for slot_idx in exclude]))
            target_slot_idxs.append(list(remaining_slot_idxs))
        # we need to have captured all target slots at this point
        assert all_target_slot_idxs == set().union(*target_slot_idxs)

        for i in range(len(cohorts)):
            cohort = cohorts[i]
            # segment the cohort into sublists that contain either a single NOS function call or no NOS function calls
            # (i.e., only computed cols)
            assert len(cohort) > 0
            # create the first segment here, so we can avoid checking for an empty list in the loop
            segments = [[cohort[0]]]
            is_nos_segment = self._is_nos_call(cohort[0])
            nos_fn: Optional[func.NOSFunction] = self._get_nos_fn(cohort[0])
            for e in cohort[1:]:
                if self._is_nos_call(e):
                    segments.append([e])
                    is_nos_segment = True
                    nos_fn = self._get_nos_fn(e)
                else:
                    if is_nos_segment:
                        # start a new segment
                        segments.append([])
                        is_nos_segment = False
                    segments[-1].append(e)

            # we create the EvalCtxs manually because create_eval_ctx() would repeat the dependencies of each segment
            segment_ctxs = [
                exprs.RowBuilder.EvalCtx(
                    slot_idxs=[e.slot_idx for e in s], exprs=s, target_slot_idxs=[], target_exprs=[])
                for s in segments
            ]
            cohort_info = self.Cohort(cohort, nos_fn, segment_ctxs, target_slot_idxs[i])
            self.cohorts.append(cohort_info)

    def _exec_cohort(self, cohort: Cohort, rows: DataRowBatch) -> None:
        """Compute the cohort for the entire input batch by dividing it up into sub-batches"""
        batch_start_idx = 0  # start row of the current sub-batch
        # for multi-resolution models, we re-assess the correct NOS batch size for each input batch
        #verify_nos_batch_size = cohort.nos_function is not None and cohort.nos_function.is_multi_res_model()
        nos_batch_size = cohort.nos_function.batch_size if cohort.nos_function is not None else None

        while batch_start_idx < len(rows):
            num_batch_rows = min(cohort.batch_size, len(rows) - batch_start_idx)
            for segment_ctx in cohort.segment_ctxs:
                if not self._is_nos_call(segment_ctx.exprs[0]):
                    # compute batch row-wise
                    for row_idx in range(batch_start_idx, batch_start_idx + num_batch_rows):
                        self.row_builder.eval(
                            rows[row_idx], segment_ctx, self.ctx.profile, ignore_errors=self.ignore_errors)
                else:
                    fn_call = segment_ctx.exprs[0]
                    # make a batched NOS call
                    arg_batches = [[] for _ in range(len(fn_call.args))]

                    valid_batch_idxs: List[int] = []  # rows with exceptions are not valid
                    for row_idx in range(batch_start_idx, batch_start_idx + num_batch_rows):
                        row = rows[row_idx]
                        if row.has_exc(fn_call.slot_idx):
                            # one of our inputs had an exception, skip this row
                            continue
                        valid_batch_idxs.append(row_idx)
                        args, kwargs = fn_call._make_args(row)
                        assert len(kwargs) == 0
                        for i in range(len(args)):
                            arg_batches[i].append(args[i])
                    num_valid_batch_rows = len(valid_batch_idxs)

                    if nos_batch_size is None:
                        # we need to choose a batch size based on the args
                        sample_args = [arg_batches[i][0] for i in range(len(arg_batches))]
                        nos_batch_size = fn_call.fn.get_batch_size(*sample_args)

                    num_remaining_batch_rows = num_valid_batch_rows
                    while num_remaining_batch_rows > 0:
                        # we make NOS calls in batches of nos_batch_size
                        if nos_batch_size is None:
                            pass
                        num_nos_batch_rows = min(nos_batch_size, num_remaining_batch_rows)
                        nos_batch_offset = num_valid_batch_rows - num_remaining_batch_rows  # offset into args, not rows
                        call_args = [
                            arg_batches[i][nos_batch_offset:nos_batch_offset + num_nos_batch_rows]
                            for i in range(len(arg_batches))
                        ]
                        start_ts = time.perf_counter()
                        result_batch = fn_call.fn.invoke(call_args, {})
                        self.ctx.profile.eval_time[fn_call.slot_idx] += time.perf_counter() - start_ts
                        self.ctx.profile.eval_count[fn_call.slot_idx] += num_nos_batch_rows

                        # move the result into the row batch
                        for result_idx in range(len(result_batch)):
                            row_idx = valid_batch_idxs[nos_batch_offset + result_idx]
                            row = rows[row_idx]
                            row[fn_call.slot_idx] = result_batch[result_idx]

                        num_remaining_batch_rows -= num_nos_batch_rows

                    # switch to the NOS-recommended batch size
                    cohort.batch_size = nos_batch_size

            # make sure images for stored cols have been saved to files before moving on to the next batch
            rows.flush_imgs(
                slice(batch_start_idx, batch_start_idx + num_batch_rows), self.stored_img_cols, self.flushed_img_slots)
            if self.pbar is not None:
                self.pbar.update(num_batch_rows * len(cohort.target_slot_idxs))
            batch_start_idx += num_batch_rows

