from __future__ import annotations

import datetime
import glob
import http.server
import importlib
import importlib.util
import inspect
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
import uuid
import warnings
from abc import abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from sys import stdout
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional, TypeVar
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pixeltable_pgserver
import sqlalchemy as sql
from tqdm import TqdmWarning

from pixeltable import exceptions as excs
from pixeltable.config import Config
from pixeltable.utils.console_output import ConsoleLogger, ConsoleMessageFilter, ConsoleOutputHandler, map_level
from pixeltable.utils.http_server import make_server

if TYPE_CHECKING:
    import spacy


_logger = logging.getLogger('pixeltable')

T = TypeVar('T')


class Env:
    """
    Store for runtime globals.
    """

    _instance: Optional[Env] = None
    _log_fmt_str = '%(asctime)s %(levelname)s %(name)s %(filename)s:%(lineno)d: %(message)s'

    _media_dir: Optional[Path]
    _file_cache_dir: Optional[Path]  # cached media files with external URL
    _dataset_cache_dir: Optional[Path]  # cached datasets (eg, pytorch or COCO)
    _log_dir: Optional[Path]  # log files
    _tmp_dir: Optional[Path]  # any tmp files
    _sa_engine: Optional[sql.engine.base.Engine]
    _pgdata_dir: Optional[Path]
    _db_name: Optional[str]
    _db_server: Optional[pixeltable_pgserver.PostgresServer]
    _db_url: Optional[str]
    _default_time_zone: Optional[ZoneInfo]

    # info about optional packages that are utilized by some parts of the code
    __optional_packages: dict[str, PackageInfo]

    _spacy_nlp: Optional[spacy.Language]
    _httpd: Optional[http.server.HTTPServer]
    _http_address: Optional[str]
    _logger: logging.Logger
    _console_logger: ConsoleLogger
    _default_log_level: int
    _logfilename: Optional[str]
    _log_to_stdout: bool
    _module_log_level: dict[str, int]  # module name -> log level
    _file_cache_size_g: float
    _pxt_api_key: Optional[str]
    _stdout_handler: logging.StreamHandler
    _initialized: bool

    _resource_pool_info: dict[str, Any]
    _current_conn: Optional[sql.Connection]
    _current_session: Optional[sql.orm.Session]

    @classmethod
    def get(cls) -> Env:
        if cls._instance is None:
            cls._init_env()
        return cls._instance

    @classmethod
    def _init_env(cls, reinit_db: bool = False) -> None:
        env = Env()
        env._set_up(reinit_db=reinit_db)
        env._upgrade_metadata()
        cls._instance = env

    def __init__(self):
        assert self._instance is None, 'Env is a singleton; use Env.get() to access the instance'

        self._media_dir = None  # computed media files
        self._file_cache_dir = None  # cached media files with external URL
        self._dataset_cache_dir = None  # cached datasets (eg, pytorch or COCO)
        self._log_dir = None  # log files
        self._tmp_dir = None  # any tmp files
        self._sa_engine = None
        self._pgdata_dir = None
        self._db_name = None
        self._db_server = None
        self._db_url = None
        self._default_time_zone = None

        self.__optional_packages = {}
        self._spacy_nlp = None
        self._httpd = None
        self._http_address = None

        # logging-related state
        self._logger = logging.getLogger('pixeltable')
        self._logger.setLevel(logging.DEBUG)  # allow everything to pass, we filter in _log_filter()
        self._logger.propagate = False
        self._logger.addFilter(self._log_filter)
        self._default_log_level = logging.INFO
        self._logfilename = None
        self._log_to_stdout = False
        self._module_log_level = {}  # module name -> log level

        # create logging handler to also log to stdout
        self._stdout_handler = logging.StreamHandler(stream=sys.stdout)
        self._stdout_handler.setFormatter(logging.Formatter(self._log_fmt_str))
        self._initialized = False

        self._resource_pool_info = {}
        self._current_conn = None
        self._current_session = None

    @property
    def db_url(self) -> str:
        assert self._db_url is not None
        return self._db_url

    @property
    def http_address(self) -> str:
        assert self._http_address is not None
        return self._http_address

    @property
    def default_time_zone(self) -> Optional[ZoneInfo]:
        return self._default_time_zone

    @default_time_zone.setter
    def default_time_zone(self, tz: Optional[ZoneInfo]) -> None:
        """
        This is not a publicly visible setter; it is only for testing purposes.
        """
        tz_name = None if tz is None else tz.key
        self.engine.dispose()
        self._create_engine(time_zone_name=tz_name)

    @property
    def conn(self) -> Optional[sql.Connection]:
        assert self._current_conn is not None
        return self._current_conn

    @property
    def session(self) -> Optional[sql.orm.Session]:
        assert self._current_session is not None
        return self._current_session

    @contextmanager
    def begin_xact(self) -> Iterator[sql.Connection]:
        """Return a context manager that yields a connection to the database. Idempotent."""
        if self._current_conn is None:
            assert self._current_session is None
            with self.engine.begin() as conn, sql.orm.Session(conn) as session:
                self._current_conn = conn
                self._current_session = session
                try:
                    yield conn
                finally:
                    self._current_session = None
                    self._current_conn = None
        else:
            assert self._current_session is not None
            yield self._current_conn

    def configure_logging(
        self,
        *,
        to_stdout: Optional[bool] = None,
        level: Optional[int] = None,
        add: Optional[str] = None,
        remove: Optional[str] = None,
    ) -> None:
        """Configure logging.

        Args:
            to_stdout: if True, also log to stdout
            level: default log level
            add: comma-separated list of 'module name:log level' pairs; ex.: add='video:10'
            remove: comma-separated list of module names
        """
        if to_stdout is not None:
            self.log_to_stdout(to_stdout)
        if level is not None:
            self.set_log_level(level)
        if add is not None:
            for module, level_str in [t.split(':') for t in add.split(',')]:
                self.set_module_log_level(module, int(level_str))
        if remove is not None:
            for module in remove.split(','):
                self.set_module_log_level(module, None)
        if to_stdout is None and level is None and add is None and remove is None:
            self.print_log_config()

    def print_log_config(self) -> None:
        print(f'logging to {self._logfilename}')
        print(f'{"" if self._log_to_stdout else "not "}logging to stdout')
        print(f'default log level: {logging.getLevelName(self._default_log_level)}')
        print(
            f'module log levels: '
            f'{",".join([name + ":" + logging.getLevelName(val) for name, val in self._module_log_level.items()])}'
        )

    def log_to_stdout(self, enable: bool = True) -> None:
        self._log_to_stdout = enable
        if enable:
            self._logger.addHandler(self._stdout_handler)
        else:
            self._logger.removeHandler(self._stdout_handler)

    def set_log_level(self, level: int) -> None:
        self._default_log_level = level

    def set_module_log_level(self, module: str, level: Optional[int]) -> None:
        if level is None:
            self._module_log_level.pop(module, None)
        else:
            self._module_log_level[module] = level

    def is_installed_package(self, package_name: str) -> bool:
        assert package_name in self.__optional_packages
        return self.__optional_packages[package_name].is_installed

    def _log_filter(self, record: logging.LogRecord) -> bool:
        if record.name == 'pixeltable':
            # accept log messages from a configured pixeltable module (at any level of the module hierarchy)
            path_parts = list(Path(record.pathname).parts)
            path_parts.reverse()
            max_idx = path_parts.index('pixeltable')
            for module_name in path_parts[:max_idx]:
                if module_name in self._module_log_level and record.levelno >= self._module_log_level[module_name]:
                    return True
        return record.levelno >= self._default_log_level

    @property
    def console_logger(self) -> ConsoleLogger:
        return self._console_logger

    def _set_up(self, echo: bool = False, reinit_db: bool = False) -> None:
        if self._initialized:
            return

        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        config = Config.get()

        self._initialized = True
        self._media_dir = Config.get().home / 'media'
        self._file_cache_dir = Config.get().home / 'file_cache'
        self._dataset_cache_dir = Config.get().home / 'dataset_cache'
        self._log_dir = Config.get().home / 'logs'
        self._tmp_dir = Config.get().home / 'tmp'

        if not self._media_dir.exists():
            self._media_dir.mkdir()
        if not self._file_cache_dir.exists():
            self._file_cache_dir.mkdir()
        if not self._dataset_cache_dir.exists():
            self._dataset_cache_dir.mkdir()
        if not self._log_dir.exists():
            self._log_dir.mkdir()
        if not self._tmp_dir.exists():
            self._tmp_dir.mkdir()

        self._file_cache_size_g = config.get_float_value('file_cache_size_g')
        if self._file_cache_size_g is None:
            raise excs.Error(
                'pixeltable/file_cache_size_g is missing from configuration\n'
                f'(either add a `file_cache_size_g` entry to the `pixeltable` section of {Config.get().config_file},\n'
                'or set the PIXELTABLE_FILE_CACHE_SIZE_G environment variable)'
            )
        self._pxt_api_key = config.get_string_value('api_key')

        # Disable spurious warnings
        warnings.simplefilter('ignore', category=TqdmWarning)
        if config.get_bool_value('hide_warnings'):
            # Disable more warnings
            warnings.simplefilter('ignore', category=UserWarning)
            warnings.simplefilter('ignore', category=FutureWarning)

        # Set verbose level for user visible console messages
        verbosity = map_level(config.get_int_value('verbosity'))
        stdout_handler = ConsoleOutputHandler(stream=stdout)
        stdout_handler.setLevel(verbosity)
        stdout_handler.addFilter(ConsoleMessageFilter())
        self._logger.addHandler(stdout_handler)
        self._console_logger = ConsoleLogger(self._logger)

        # configure _logger to log to a file
        self._logfilename = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.log'
        fh = logging.FileHandler(self._log_dir / self._logfilename, mode='w')
        fh.setFormatter(logging.Formatter(self._log_fmt_str))
        self._logger.addHandler(fh)

        # configure sqlalchemy logging
        sql_logger = logging.getLogger('sqlalchemy.engine')
        sql_logger.setLevel(logging.INFO)
        sql_logger.addHandler(fh)
        sql_logger.propagate = False

        # configure pyav logging
        av_logfilename = self._logfilename.replace('.log', '_av.log')
        av_fh = logging.FileHandler(self._log_dir / av_logfilename, mode='w')
        av_fh.setFormatter(logging.Formatter(self._log_fmt_str))
        av_logger = logging.getLogger('libav')
        av_logger.addHandler(av_fh)
        av_logger.propagate = False

        # configure web-server logging
        http_logfilename = self._logfilename.replace('.log', '_http.log')
        http_fh = logging.FileHandler(self._log_dir / http_logfilename, mode='w')
        http_fh.setFormatter(logging.Formatter(self._log_fmt_str))
        http_logger = logging.getLogger('pixeltable.http.server')
        http_logger.addHandler(http_fh)
        http_logger.propagate = False

        self.clear_tmp_dir()

        self._db_name = os.environ.get('PIXELTABLE_DB', 'pixeltable')
        self._pgdata_dir = Path(os.environ.get('PIXELTABLE_PGDATA', str(Config.get().home / 'pgdata')))

        # cleanup_mode=None will leave the postgres process running after Python exits
        # cleanup_mode='stop' will terminate the postgres process when Python exits
        # On Windows, we need cleanup_mode='stop' because child processes are killed automatically when the parent
        # process (such as Terminal or VSCode) exits, potentially leaving it in an unusable state.
        cleanup_mode = 'stop' if platform.system() == 'Windows' else None
        self._db_server = pixeltable_pgserver.get_server(self._pgdata_dir, cleanup_mode=cleanup_mode)
        self._db_url = self._db_server.get_uri(database=self._db_name, driver='psycopg')

        tz_name = config.get_string_value('time_zone')
        if tz_name is not None:
            # Validate tzname
            if not isinstance(tz_name, str):
                self._logger.error('Invalid time zone specified in configuration.')
            else:
                try:
                    _ = ZoneInfo(tz_name)
                except ZoneInfoNotFoundError:
                    self._logger.error(f'Invalid time zone specified in configuration: {tz_name}')

        if reinit_db and self._store_db_exists():
            self._drop_store_db()

        create_db = not self._store_db_exists()

        if create_db:
            self._logger.info(f'creating database at: {self.db_url}')
            self._create_store_db()
        else:
            self._logger.info(f'found database at: {self.db_url}')

        # Create the SQLAlchemy engine. This will also set the default time zone.
        self._create_engine(time_zone_name=tz_name, echo=echo)

        if create_db:
            from pixeltable import metadata

            metadata.schema.base_metadata.create_all(self._sa_engine)
            metadata.create_system_info(self._sa_engine)

        self.console_logger.info(f'Connected to Pixeltable database at: {self.db_url}')

        # we now have a home directory and db; start other services
        self._set_up_runtime()
        self.log_to_stdout(False)

    def _create_engine(self, time_zone_name: Optional[str], echo: bool = False) -> None:
        connect_args = {} if time_zone_name is None else {'options': f'-c timezone={time_zone_name}'}
        self._sa_engine = sql.create_engine(
            self.db_url, echo=echo, future=True, isolation_level='REPEATABLE READ', connect_args=connect_args
        )
        self._logger.info(f'Created SQLAlchemy engine at: {self.db_url}')
        with self.engine.begin() as conn:
            tz_name = conn.execute(sql.text('SHOW TIME ZONE')).scalar()
            assert isinstance(tz_name, str)
            self._logger.info(f'Database time zone is now: {tz_name}')
            self._default_time_zone = ZoneInfo(tz_name)

    def _store_db_exists(self) -> bool:
        assert self._db_name is not None
        # don't try to connect to self.db_name, it may not exist
        db_url = self._db_server.get_uri(database='postgres', driver='psycopg')
        engine = sql.create_engine(db_url, future=True)
        try:
            with engine.begin() as conn:
                stmt = f"SELECT COUNT(*) FROM pg_database WHERE datname = '{self._db_name}'"
                result = conn.scalar(sql.text(stmt))
                assert result <= 1
                return result == 1
        finally:
            engine.dispose()

    def _create_store_db(self) -> None:
        assert self._db_name is not None
        # create the db
        pg_db_url = self._db_server.get_uri(database='postgres', driver='psycopg')
        engine = sql.create_engine(pg_db_url, future=True, isolation_level='AUTOCOMMIT')
        preparer = engine.dialect.identifier_preparer
        try:
            with engine.begin() as conn:
                # use C collation to get standard C/Python-style sorting
                stmt = (
                    f'CREATE DATABASE {preparer.quote(self._db_name)} '
                    "ENCODING 'utf-8' LC_COLLATE 'C' LC_CTYPE 'C' TEMPLATE template0"
                )
                conn.execute(sql.text(stmt))
        finally:
            engine.dispose()

        # enable pgvector
        store_db_url = self._db_server.get_uri(database=self._db_name, driver='psycopg')
        engine = sql.create_engine(store_db_url, future=True, isolation_level='AUTOCOMMIT')
        try:
            with engine.begin() as conn:
                conn.execute(sql.text('CREATE EXTENSION vector'))
        finally:
            engine.dispose()

    def _drop_store_db(self) -> None:
        assert self._db_name is not None
        db_url = self._db_server.get_uri(database='postgres', driver='psycopg')
        engine = sql.create_engine(db_url, future=True, isolation_level='AUTOCOMMIT')
        preparer = engine.dialect.identifier_preparer
        try:
            with engine.begin() as conn:
                # terminate active connections
                stmt = f"""
                    SELECT pg_terminate_backend(pg_stat_activity.pid)
                    FROM pg_stat_activity
                    WHERE pg_stat_activity.datname = '{self._db_name}'
                    AND pid <> pg_backend_pid()
                """
                conn.execute(sql.text(stmt))
                # drop db
                stmt = f'DROP DATABASE {preparer.quote(self._db_name)}'
                conn.execute(sql.text(stmt))
        finally:
            engine.dispose()

    def _upgrade_metadata(self) -> None:
        from pixeltable import metadata

        metadata.upgrade_md(self._sa_engine)

    @property
    def pxt_api_key(self) -> str:
        if self._pxt_api_key is None:
            raise excs.Error(
                'No API key is configured. Set the PIXELTABLE_API_KEY environment variable, or add an entry to '
                'config.toml as described here:\nhttps://pixeltable.github.io/pixeltable/config/'
            )
        return self._pxt_api_key

    def get_client(self, name: str) -> Any:
        """
        Gets the client with the specified name, initializing it if necessary.

        Args:
            - name: The name of the client
        """
        cl = _registered_clients[name]
        if cl.client_obj is not None:
            return cl.client_obj  # Already initialized

        # Construct a client, retrieving each parameter from config.

        init_kwargs: dict[str, str] = {}
        for param in cl.param_names:
            arg = Config.get().get_string_value(param, section=name)
            if arg is not None and len(arg) > 0:
                init_kwargs[param] = arg
            else:
                raise excs.Error(
                    f'`{name}` client not initialized: parameter `{param}` is not configured.\n'
                    f'To fix this, specify the `{name.upper()}_{param.upper()}` environment variable, '
                    f'or put `{param.lower()}` in the `{name.lower()}` section of $PIXELTABLE_HOME/config.toml.'
                )

        cl.client_obj = cl.init_fn(**init_kwargs)
        self._logger.info(f'Initialized `{name}` client.')
        return cl.client_obj

    def _start_web_server(self) -> None:
        """
        The http server root is the file system root.
        eg: /home/media/foo.mp4 is located at http://127.0.0.1:{port}/home/media/foo.mp4
        in windows, the server will translate paths like http://127.0.0.1:{port}/c:/media/foo.mp4
        This arrangement enables serving media hosted within _home,
        as well as external media inserted into pixeltable or produced by pixeltable.
        The port is chosen dynamically to prevent conflicts.
        """
        # Port 0 means OS picks one for us.
        self._httpd = make_server('127.0.0.1', 0)
        port = self._httpd.server_address[1]
        self._http_address = f'http://127.0.0.1:{port}'

        def run_server():
            logging.log(logging.INFO, f'running web server at {self._http_address}')
            self._httpd.serve_forever()

        # Run the server in a separate thread
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()

    def _set_up_runtime(self) -> None:
        """Check for and start runtime services"""
        self._start_web_server()
        self.__register_packages()
        if self.is_installed_package('spacy'):
            self.__init_spacy()

    def __register_packages(self) -> None:
        """Declare optional packages that are utilized by some parts of the code."""
        self.__register_package('anthropic')
        self.__register_package('boto3')
        self.__register_package('datasets')
        self.__register_package('fiftyone')
        self.__register_package('fireworks', library_name='fireworks-ai')
        self.__register_package('google.generativeai', library_name='google-generativeai')
        self.__register_package('huggingface_hub', library_name='huggingface-hub')
        self.__register_package('label_studio_sdk', library_name='label-studio-sdk')
        self.__register_package('llama_cpp', library_name='llama-cpp-python')
        self.__register_package('mistralai')
        self.__register_package('mistune')
        self.__register_package('ollama')
        self.__register_package('openai')
        self.__register_package('openpyxl')
        self.__register_package('pyarrow')
        self.__register_package('pydantic')
        self.__register_package('replicate')
        self.__register_package('sentencepiece')
        self.__register_package('sentence_transformers', library_name='sentence-transformers')
        self.__register_package('spacy')
        self.__register_package('tiktoken')
        self.__register_package('together')
        self.__register_package('torch')
        self.__register_package('torchaudio')
        self.__register_package('torchvision')
        self.__register_package('transformers')
        self.__register_package('whisper', library_name='openai-whisper')
        self.__register_package('whisperx')
        self.__register_package('yolox', library_name='git+https://github.com/Megvii-BaseDetection/YOLOX@ac58e0a')

    def __register_package(self, package_name: str, library_name: Optional[str] = None) -> None:
        is_installed: bool
        try:
            is_installed = importlib.util.find_spec(package_name) is not None
        except ModuleNotFoundError:
            # This can happen if the parent of `package_name` is not installed.
            is_installed = False
        self.__optional_packages[package_name] = PackageInfo(
            is_installed=is_installed,
            library_name=library_name or package_name,  # defaults to package_name unless specified otherwise
        )

    def require_package(self, package_name: str, min_version: Optional[list[int]] = None) -> None:
        """
        Checks whether the specified optional package is available. If not, raises an exception
        with an error message informing the user how to install it.
        """
        assert package_name in self.__optional_packages
        package_info = self.__optional_packages[package_name]

        if not package_info.is_installed:
            # Check again whether the package has been installed.
            # We do this so that if a user gets an "optional library not found" error message, they can
            # `pip install` the library and re-run the Pixeltable operation without having to restart
            # their Python session.
            package_info.is_installed = importlib.util.find_spec(package_name) is not None
            if not package_info.is_installed:
                # Still not found.
                raise excs.Error(
                    f'This feature requires the `{package_name}` package. To install it, run: '
                    f'`pip install -U {package_info.library_name}`'
                )

        if min_version is None:
            return

        # check whether we have a version >= the required one
        if package_info.version is None:
            module = importlib.import_module(package_name)
            package_info.version = [int(x) for x in module.__version__.split('.')]

        if min_version > package_info.version:
            raise excs.Error(
                f'The installed version of package `{package_name}` is '
                f'{".".join(str(v) for v in package_info.version)}, '
                f'but version >={".".join(str(v) for v in min_version)} is required. '
                f'To fix this, run: `pip install -U {package_info.library_name}`'
            )

    def __init_spacy(self) -> None:
        """
        spaCy relies on a pip-installed model to operate. In order to avoid requiring the model as a separate
        dependency, we install it programmatically here. This should cause no problems, since the model packages
        have no sub-dependencies (in fact, this is how spaCy normally manages its model resources).
        """
        import spacy
        from spacy.cli.download import get_model_filename

        spacy_model = 'en_core_web_sm'
        spacy_model_version = '3.7.1'
        filename = get_model_filename(spacy_model, spacy_model_version, sdist=False)
        url = f'{spacy.about.__download_url__}/{filename}'
        # Try to `pip install` the model. We set check=False; if the pip command fails, it's not necessarily
        # a problem, because the model have been installed on a previous attempt.
        self._logger.info(f'Ensuring spaCy model is installed: {filename}')
        ret = subprocess.run([sys.executable, '-m', 'pip', 'install', '-qU', url], check=False)
        if ret.returncode != 0:
            self._logger.warning(f'pip install failed for spaCy model: {filename}')
        try:
            self._logger.info(f'Loading spaCy model: {spacy_model}')
            self._spacy_nlp = spacy.load(spacy_model)
        except Exception as exc:
            self._logger.warning(f'Failed to load spaCy model: {spacy_model}', exc_info=exc)
            warnings.warn(
                f"Failed to load spaCy model '{spacy_model}'. spaCy features will not be available.",
                excs.PixeltableWarning,
                stacklevel=1,
            )
            self.__optional_packages['spacy'].is_installed = False

    def clear_tmp_dir(self) -> None:
        for path in glob.glob(f'{self._tmp_dir}/*'):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

    def num_tmp_files(self) -> int:
        return len(glob.glob(f'{self._tmp_dir}/*'))

    def create_tmp_path(self, extension: str = '') -> Path:
        return self._tmp_dir / f'{uuid.uuid4()}{extension}'

    # def get_resource_pool_info(self, pool_id: str, pool_info_cls: Optional[Type[T]]) -> T:
    def get_resource_pool_info(self, pool_id: str, make_pool_info: Optional[Callable[[], T]] = None) -> T:
        """Returns the info object for the given id, creating it if necessary."""
        info = self._resource_pool_info.get(pool_id)
        if info is None and make_pool_info is not None:
            info = make_pool_info()
            self._resource_pool_info[pool_id] = info
        return info

    @property
    def media_dir(self) -> Path:
        assert self._media_dir is not None
        return self._media_dir

    @property
    def file_cache_dir(self) -> Path:
        assert self._file_cache_dir is not None
        return self._file_cache_dir

    @property
    def dataset_cache_dir(self) -> Path:
        assert self._dataset_cache_dir is not None
        return self._dataset_cache_dir

    @property
    def tmp_dir(self) -> Path:
        assert self._tmp_dir is not None
        return self._tmp_dir

    @property
    def engine(self) -> sql.engine.base.Engine:
        assert self._sa_engine is not None
        return self._sa_engine

    @property
    def spacy_nlp(self) -> spacy.Language:
        Env.get().require_package('spacy')
        assert self._spacy_nlp is not None
        return self._spacy_nlp


def register_client(name: str) -> Callable:
    """Decorator that registers a third-party API client for use by Pixeltable.

    The decorated function is an initialization wrapper for the client, and can have
    any number of string parameters, with a signature such as:

    ```
    def my_client(api_key: str, url: str) -> my_client_sdk.Client:
        return my_client_sdk.Client(api_key=api_key, url=url)
    ```

    The initialization wrapper will not be called immediately; initialization will
    be deferred until the first time the client is used. At initialization time,
    Pixeltable will attempt to load the client parameters from config. For each
    config parameter:
    - If an environment variable named MY_CLIENT_API_KEY (for example) is set, use it;
    - Otherwise, look for 'api_key' in the 'my_client' section of config.toml.

    If all config parameters are found, Pixeltable calls the initialization function;
    otherwise it throws an exception.

    Args:
        - name (str): The name of the API client (e.g., 'openai' or 'label-studio').
    """

    def decorator(fn: Callable) -> None:
        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())
        _registered_clients[name] = ApiClient(init_fn=fn, param_names=param_names)

    return decorator


_registered_clients: dict[str, ApiClient] = {}


@dataclass
class ApiClient:
    init_fn: Callable
    param_names: list[str]
    client_obj: Optional[Any] = None


@dataclass
class PackageInfo:
    is_installed: bool
    library_name: str  # pypi library name (may be different from package name)
    version: Optional[list[int]] = None  # installed version, as a list of components (such as [3,0,2] for "3.0.2")


TIME_FORMAT = '%H:%M.%S %f'


@dataclass
class RateLimitsInfo:
    """
    Abstract base class for resource pools made up of rate limits for different resources.

    Rate limits and currently remaining resources are periodically reported via record().

    Subclasses provide operational customization via:
    - get_retry_delay()
    - get_request_resources(self, ...) -> dict[str, int]
    with parameters that are a subset of those of the udf that creates the subclass's instance
    """

    # get_request_resources:
    # - Returns estimated resources needed for a specific request (ie, a single udf call) as a dict (key: resource name)
    # - parameters are a subset of those of the udf
    # - this is not a class method because the signature depends on the instantiating udf
    get_request_resources: Callable[..., dict[str, int]]

    resource_limits: dict[str, RateLimitInfo] = field(default_factory=dict)

    def is_initialized(self) -> bool:
        return len(self.resource_limits) > 0

    def reset(self) -> None:
        self.resource_limits.clear()

    def record(self, **kwargs) -> None:
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        if len(self.resource_limits) == 0:
            self.resource_limits = {k: RateLimitInfo(k, now, *v) for k, v in kwargs.items() if v is not None}
            # TODO: remove
            for info in self.resource_limits.values():
                _logger.debug(
                    f'Init {info.resource} rate limit: rem={info.remaining} '
                    f'reset={info.reset_at.strftime(TIME_FORMAT)} delta={(info.reset_at - now).total_seconds()}'
                )
        else:
            for k, v in kwargs.items():
                if v is not None:
                    self.resource_limits[k].update(now, *v)

    @abstractmethod
    def get_retry_delay(self, exc: Exception) -> Optional[float]:
        """Returns number of seconds to wait before retry, or None if not retryable"""
        pass


@dataclass
class RateLimitInfo:
    """Container for rate limit-related information for a single resource."""

    resource: str
    recorded_at: datetime.datetime
    limit: int
    remaining: int
    reset_at: datetime.datetime

    def update(self, recorded_at: datetime.datetime, limit: int, remaining: int, reset_at: datetime.datetime) -> None:
        # we always update everything, even though responses may come back out-of-order: we can't use reset_at to
        # determine order, because it doesn't increase monotonically (the reeset duration shortens as output_tokens
        # are freed up - going from max to actual)
        self.recorded_at = recorded_at
        self.limit = limit
        self.remaining = remaining
        reset_delta = reset_at - self.reset_at
        self.reset_at = reset_at
        # TODO: remove
        _logger.debug(
            f'Update {self.resource} rate limit: rem={self.remaining} reset={self.reset_at.strftime(TIME_FORMAT)} '
            f'reset_delta={reset_delta.total_seconds()} recorded_delta={(self.reset_at - recorded_at).total_seconds()}'
        )
