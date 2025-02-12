import datetime
import glob
import json
import os
import random
import urllib.parse
from pathlib import Path
from typing import Any, Hashable, Optional

import more_itertools
import numpy as np
import pandas as pd
import PIL.Image
import pytest

import pixeltable as pxt
import pixeltable.exceptions as excs
import pixeltable.utils.s3 as s3_util
from pixeltable import catalog, exprs
from pixeltable.catalog.globals import UpdateStatus
from pixeltable.dataframe import DataFrameResultSet
from pixeltable.env import Env
from pixeltable.functions.huggingface import clip, sentence_transformer
from pixeltable.io import SyncStatus


def make_default_type(t: pxt.ColumnType.Type) -> pxt.ColumnType:
    if t == pxt.ColumnType.Type.STRING:
        return pxt.StringType()
    if t == pxt.ColumnType.Type.INT:
        return pxt.IntType()
    if t == pxt.ColumnType.Type.FLOAT:
        return pxt.FloatType()
    if t == pxt.ColumnType.Type.BOOL:
        return pxt.BoolType()
    if t == pxt.ColumnType.Type.TIMESTAMP:
        return pxt.TimestampType()
    assert False


def make_tbl(name: str = 'test', col_names: Optional[list[str]] = None) -> pxt.Table:
    if col_names is None:
        col_names = ['c1']
    schema: dict[str, pxt.ColumnType] = {}
    for i, col_name in enumerate(col_names):
        schema[f'{col_name}'] = make_default_type(pxt.ColumnType.Type(i % 5))
    return pxt.create_table(name, schema)


def create_table_data(
    t: catalog.Table, col_names: Optional[list[str]] = None, num_rows: int = 10
) -> list[dict[str, Any]]:
    if col_names is None:
        col_names = []
    data: dict[str, Any] = {}

    sample_dict = {
        'detections': [
            {
                'id': '637e8e073b28441a453564cf',
                'attributes': {},
                'tags': [],
                'label': 'potted plant',
                'bounding_box': [0.37028125, 0.3345305164319249, 0.038593749999999996, 0.16314553990610328],
                'mask': None,
                'confidence': None,
                'index': None,
                'supercategory': 'furniture',
                'iscrowd': 0,
            },
            {
                'id': '637e8e073b28441a453564cf',
                'attributes': {},
                'tags': [],
                'label': 'potted plant',
                'bounding_box': [0.37028125, 0.3345305164319249, 0.038593749999999996, 0.16314553990610328],
                'mask': None,
                'confidence': None,
                'index': None,
                'supercategory': 'furniture',
                'iscrowd': 0,
            },
        ]
    }

    if len(col_names) == 0:
        col_names = [c.name for c in t._tbl_version_path.columns() if not c.is_computed]

    col_types = t._schema
    for col_name in col_names:
        col_type = col_types[col_name]
        col_data: Any = None
        if col_type.is_string_type():
            col_data = ['test string'] * num_rows
        if col_type.is_int_type():
            col_data = np.random.randint(0, 100, size=num_rows).tolist()
        if col_type.is_float_type():
            col_data = (np.random.random(size=num_rows) * 100).tolist()
        if col_type.is_bool_type():
            col_data = np.random.randint(0, 2, size=num_rows)
            col_data = [False if i == 0 else True for i in col_data]
        if col_type.is_timestamp_type():
            col_data = [datetime.datetime.now()] * num_rows
        if col_type.is_json_type():
            col_data = [sample_dict] * num_rows
        if col_type.is_array_type():
            assert isinstance(col_type, pxt.ArrayType)
            col_data = [np.ones(col_type.shape, dtype=col_type.numpy_dtype()) for i in range(num_rows)]
        if col_type.is_image_type():
            image_path = get_image_files()[0]
            col_data = [image_path for i in range(num_rows)]
        if col_type.is_video_type():
            video_path = get_video_files()[0]
            col_data = [video_path for i in range(num_rows)]
        data[col_name] = col_data
    rows = [{col_name: data[col_name][i] for col_name in col_names} for i in range(num_rows)]
    return rows


def create_test_tbl(name: str = 'test_tbl') -> catalog.Table:
    schema = {
        'c1': pxt.Required[pxt.String],
        'c1n': pxt.String,
        'c2': pxt.Required[pxt.Int],
        'c3': pxt.Required[pxt.Float],
        'c4': pxt.Required[pxt.Bool],
        'c5': pxt.Required[pxt.Timestamp],
        'c6': pxt.Required[pxt.Json],
        'c7': pxt.Required[pxt.Json],
    }
    t = pxt.create_table(name, schema, primary_key='c2')
    t.add_computed_column(c8=pxt.array([[1, 2, 3], [4, 5, 6]]))

    num_rows = 100
    d1 = {
        'f1': 'test string 1',
        'f2': 1,
        'f3': 1.0,
        'f4': True,
        'f5': [1.0, 2.0, 3.0, 4.0],
        'f6': {'f7': 'test string 2', 'f8': [1.0, 2.0, 3.0, 4.0]},
    }
    d2 = [d1, d1]

    c1_data = [f'test string {i}' for i in range(num_rows)]
    c2_data = [i for i in range(num_rows)]
    c3_data = [float(i) for i in range(num_rows)]
    c4_data = [bool(i % 2) for i in range(num_rows)]
    c5_data = [datetime.datetime(2024, 7, 1) + datetime.timedelta(hours=i) for i in range(num_rows)]
    c6_data = []
    for i in range(num_rows):
        d = {
            'f1': f'test string {i}',
            'f2': i,
            'f3': float(i),
            'f4': bool(i % 2),
            'f5': list(range(5 + i // 10)),
            #'f5': [1.0, 2.0, 3.0, 4.0],
            'f6': {'f7': 'test string 2', 'f8': [1.0, 2.0, 3.0, 4.0]},
        }
        c6_data.append(d)

    c7_data = [d2] * num_rows
    rows = [
        {
            'c1': c1_data[i],
            'c1n': c1_data[i] if i % 10 != 0 else None,
            'c2': c2_data[i],
            'c3': c3_data[i],
            'c4': c4_data[i],
            'c5': c5_data[i],
            'c6': c6_data[i],
            'c7': c7_data[i],
        }
        for i in range(num_rows)
    ]
    t.insert(rows)
    return t


def create_img_tbl(name: str = 'test_img_tbl', num_rows: int = 0) -> catalog.Table:
    schema = {'img': pxt.Required[pxt.Image], 'category': pxt.Required[pxt.String], 'split': pxt.Required[pxt.String]}
    tbl = pxt.create_table(name, schema)
    rows = read_data_file('imagenette2-160', 'manifest.csv', ['img'])
    if num_rows > 0:
        # select output_rows randomly in the hope of getting a good sample of the available categories
        rng = np.random.default_rng(17)
        idxs = rng.choice(np.arange(len(rows)), size=num_rows, replace=False)
        rows = [rows[i] for i in idxs]
    tbl.insert(rows)
    return tbl


def create_all_datatypes_tbl() -> catalog.Table:
    """Creates a table with all supported datatypes."""
    schema = {
        'row_id': pxt.Required[pxt.Int],
        'c_array': pxt.Array[(10,), pxt.Float],  # type: ignore[misc]
        'c_bool': pxt.Bool,
        'c_float': pxt.Float,
        'c_image': pxt.Image,
        'c_int': pxt.Int,
        'c_json': pxt.Json,
        'c_string': pxt.String,
        'c_timestamp': pxt.Timestamp,
        'c_video': pxt.Video,
    }
    tbl = pxt.create_table('all_datatype_tbl', schema)
    example_rows = create_table_data(tbl, num_rows=11)

    for i, r in enumerate(example_rows):
        r['row_id'] = i  # row_id

    tbl.insert(example_rows)
    return tbl


def create_scalars_tbl(num_rows: int, seed: int = 0, percent_nulls: int = 10) -> catalog.Table:
    """
    Creates a table with scalar columns, each of which contains randomly generated data.
    """
    assert percent_nulls >= 0 and percent_nulls <= 100
    rng = np.random.default_rng(seed)
    schema = {
        'row_id': pxt.IntType(nullable=False),  # used for row selection
        'c_bool': pxt.BoolType(nullable=True),
        'c_float': pxt.FloatType(nullable=True),
        'c_int': pxt.IntType(nullable=True),
        'c_string': pxt.StringType(nullable=True),
        'c_timestamp': pxt.TimestampType(nullable=True),
    }
    tbl = pxt.create_table('scalars_tbl', schema)

    example_rows: list[dict[str, Any]] = []
    str_chars = 'abcdefghijklmnopqrstuvwxyzab'
    start_date = datetime.datetime(2010, 1, 1)
    end_date = datetime.datetime(2019, 12, 31)
    delta_days = (end_date - start_date).days
    for i in range(num_rows):
        str_idx = int(rng.integers(0, 26))
        days = int(rng.integers(0, delta_days))
        seconds = int(rng.integers(0, 60 * 60 * 24))
        example_rows.append(
            {
                'row_id': i,
                'c_bool': None if rng.integers(0, 100) < percent_nulls else bool(rng.choice([True, False])),
                'c_float': None if rng.integers(0, 100) < percent_nulls else float(rng.uniform(0, 1)),
                'c_int': None if rng.integers(0, 100) < percent_nulls else int(rng.integers(0, 10)),
                'c_string': None if rng.integers(0, 100) < percent_nulls else str_chars[str_idx : str_idx + 3],
                'c_timestamp': None
                if rng.integers(0, 100) < percent_nulls
                else start_date + datetime.timedelta(days=days, seconds=seconds),
            }
        )
    tbl.insert(example_rows)
    return tbl


def read_data_file(dir_name: str, file_name: str, path_col_names: Optional[list[str]] = None) -> list[dict[str, Any]]:
    """
    Locate dir_name, create df out of file_name.
    path_col_names: col names in csv file that contain file names; those will be converted to absolute paths
    by adding the path to 'file_name' as a prefix.
    Returns:
        tuple of (list of rows, list of column names)
    """
    if path_col_names is None:
        path_col_names = []
    tests_dir = os.path.dirname(__file__)  # search with respect to tests/ dir
    glob_result = glob.glob(f'{tests_dir}/**/{dir_name}', recursive=True)
    assert len(glob_result) == 1, f'Could not find {dir_name}'
    abs_path = Path(glob_result[0])
    data_file_path = abs_path / file_name
    assert data_file_path.is_file(), f'Not a file: {str(data_file_path)}'
    df = pd.read_csv(str(data_file_path))
    for col_name in path_col_names:
        assert col_name in df.columns
        df[col_name] = df.apply(lambda r: str(abs_path / r[col_name]), axis=1)
    return df.to_dict(orient='records')  # type: ignore[return-value]


def get_video_files(include_bad_video: bool = False) -> list[str]:
    tests_dir = os.path.dirname(__file__)  # search with respect to tests/ dir
    glob_result = glob.glob(f'{tests_dir}/**/videos/*', recursive=True)
    if not include_bad_video:
        glob_result = [f for f in glob_result if 'bad_video' not in f]

    half_res = [f for f in glob_result if 'half_res' in f or 'bad_video' in f]
    half_res.sort()
    return half_res


def get_test_video_files() -> list[str]:
    tests_dir = os.path.dirname(__file__)  # search with respect to tests/ dir
    glob_result = glob.glob(f'{tests_dir}/**/test_videos/*', recursive=True)
    return glob_result


__IMAGE_FILES: list[str] = []
__IMAGE_FILES_WITH_BAD_IMAGE: list[str] = []


# Gets all image files in the test folder.
# The images will be returned in an order that: (1) is deterministic; (2) ensures that images
# of different modes appear early in the list; and (3) is appropriately randomized subject to
# these constraints.
def get_image_files(include_bad_image: bool = False) -> list[str]:
    global __IMAGE_FILES, __IMAGE_FILES_WITH_BAD_IMAGE
    if not __IMAGE_FILES:
        tests_dir = os.path.dirname(__file__)  # search with respect to tests/ dir
        img_files_path = Path(tests_dir) / 'data' / 'imagenette2-160'
        glob_result = glob.glob(f'{img_files_path}/*.JPEG')
        assert len(glob_result) > 1000
        bad_image = next(f for f in glob_result if 'bad_image' in f)
        good_images = [(__image_mode(f), f) for f in glob_result if 'bad_image' not in f]
        # Group images by mode
        modes = {mode for mode, _ in good_images}
        groups = [[f for mode, f in good_images if mode == mode_group] for mode_group in modes]
        # Sort and randomize the images in each group to ensure that the ordering is both
        # deterministic and not dependent on extrinsic characteristics such as filename
        for group in groups:
            group.sort()
            random.Random(4171780).shuffle(group)
        # Combine the groups in round-robin fashion to ensure that small initial segments
        # contain representatives from each group
        __IMAGE_FILES = list(more_itertools.roundrobin(*groups))
        __IMAGE_FILES_WITH_BAD_IMAGE = [bad_image] + __IMAGE_FILES
    return __IMAGE_FILES_WITH_BAD_IMAGE if include_bad_image else __IMAGE_FILES


def __image_mode(path: str) -> str:
    image = PIL.Image.open(path)
    try:
        return image.mode
    finally:
        image.close()


def get_multimedia_commons_video_uris(n: int = 10) -> list[str]:
    uri = 's3://multimedia-commons/data/videos/mp4/'
    parsed = urllib.parse.urlparse(uri)
    bucket_name = parsed.netloc
    prefix = parsed.path.lstrip('/')
    s3_client = s3_util.get_client()
    uris: list[str] = []
    # Use paginator to handle more than 1000 objects
    paginator = s3_client.get_paginator('list_objects_v2')

    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if 'Contents' not in page:
            continue
        for obj in page['Contents']:
            if len(uris) >= n:
                return uris
            uri = f's3://{bucket_name}/{obj["Key"]}'
            uris.append(uri)
    return uris


def get_audio_files(include_bad_audio: bool = False) -> list[str]:
    tests_dir = Path(os.path.dirname(__file__))
    audio_dir = tests_dir / 'data' / 'audio'
    glob_result = glob.glob(f'{audio_dir}/*', recursive=True)
    if not include_bad_audio:
        glob_result = [f for f in glob_result if 'bad_audio' not in f]
    return glob_result


def get_audio_file(name: str) -> Optional[str]:
    tests_dir = Path(os.path.dirname(__file__))
    audio_dir = tests_dir / 'data' / 'audio'
    file_path = audio_dir / name
    glob_result = glob.glob(f'{file_path}', recursive=True)
    return glob_result.pop(0) if len(glob_result) > 0 else None


def get_documents() -> list[str]:
    tests_dir = Path(os.path.dirname(__file__))
    docs_dir = tests_dir / 'data' / 'documents'
    return glob.glob(f'{docs_dir}/*', recursive=True)


def get_sentences(n: int = 100) -> list[str]:
    tests_dir = os.path.dirname(__file__)
    path = glob.glob(f'{tests_dir}/**/jeopardy.json', recursive=True)[0]
    with open(path, 'r', encoding='utf8') as f:
        questions_list = json.load(f)
    # this dataset contains \' around the questions
    return [q['question'].replace("'", '') for q in questions_list[:n]]


def assert_resultset_eq(r1: DataFrameResultSet, r2: DataFrameResultSet, compare_col_names: bool = False) -> None:
    assert len(r1) == len(r2)
    assert len(r1.schema) == len(r2.schema)
    assert all(type1.matches(type2) for type1, type2 in zip(r1.schema.values(), r2.schema.values()))
    if compare_col_names:
        assert r1.schema.keys() == r2.schema.keys()
    for r1_col, r2_col in zip(r1.schema, r2.schema):
        # compare column values
        s1 = r1[r1_col]
        s2 = r2[r2_col]
        if r1.schema[r1_col].is_float_type():
            # To handle None values in float columns, we need to set the dtype and allow NaN comparison
            assert np.allclose(np.array(s1, dtype=float), np.array(s2, dtype=float), equal_nan=True)
        elif r1.schema[r1_col].is_array_type():
            assert all(np.array_equal(a1, a2) for a1, a2 in zip(s1, s2))
        else:
            assert s1 == s2


def strip_lines(s: str) -> str:
    lines = s.split('\n')
    return '\n'.join(line.strip() for line in lines)


def skip_test_if_not_installed(package) -> None:
    if not Env.get().is_installed_package(package):
        pytest.skip(f'Package `{package}` is not installed.')


def skip_test_if_no_client(client_name: str) -> None:
    try:
        _ = Env.get().get_client(client_name)
    except excs.Error as exc:
        pytest.skip(str(exc))


def validate_update_status(status: UpdateStatus, expected_rows: Optional[int] = None) -> None:
    assert status.num_excs == 0
    if expected_rows is not None:
        assert status.num_rows == expected_rows, status


def validate_sync_status(
    status: SyncStatus,
    expected_external_rows_created: Optional[int] = None,
    expected_external_rows_updated: Optional[int] = None,
    expected_external_rows_deleted: Optional[int] = None,
    expected_pxt_rows_updated: Optional[int] = None,
    expected_num_excs: Optional[int] = 0,
) -> None:
    if expected_external_rows_created is not None:
        assert status.external_rows_created == expected_external_rows_created, status
    if expected_external_rows_updated is not None:
        assert status.external_rows_updated == expected_external_rows_updated, status
    if expected_external_rows_deleted is not None:
        assert status.external_rows_deleted == expected_external_rows_deleted, status
    if expected_pxt_rows_updated is not None:
        assert status.pxt_rows_updated == expected_pxt_rows_updated, status
    if expected_num_excs is not None:
        assert status.num_excs == expected_num_excs, status


def make_test_arrow_table(output_path: Path) -> str:
    import pyarrow as pa
    from pyarrow import parquet

    float_array = [[1.0, 2.0], [10.0, 20.0], [100.0, 200.0], [1000.0, 2000.0], [10000.0, 20000.0]]
    value_dict: dict[str, list] = {
        'c_id': [1, 2, 3, 4, 5],
        'c_int64': [-10, -20, -30, -40, None],
        'c_int32': [-1, -2, -3, -4, None],
        'c_float32': [1.1, 2.2, 3.3, 4.4, None],
        'c_string': ['aaa', 'bbb', 'ccc', 'ddd', None],
        'c_boolean': [True, False, True, False, None],
        'c_timestamp': [
            datetime.datetime(2012, 1, 1, 12, 0, 0, 25),
            datetime.datetime(2012, 1, 2, 12, 0, 0, 25),
            datetime.datetime(2012, 1, 3, 12, 0, 0, 25),
            datetime.datetime(2012, 1, 4, 12, 0, 0, 25),
            None,
        ],
        # The pyarrow fixed_shape_tensor type does not support NULLs (currently can write them but not read them)
        # So, no nulls in this column
        'c_array_float32': float_array,
    }

    arr_size = len(float_array[0])
    tensor_type = pa.fixed_shape_tensor(pa.float32(), (arr_size,))

    fields = [
        ('c_id', pa.int32()),
        ('c_int64', pa.int64()),
        ('c_int32', pa.int32()),
        ('c_float32', pa.float32()),
        ('c_string', pa.string()),
        ('c_boolean', pa.bool_()),
        ('c_timestamp', pa.timestamp('us')),
        ('c_array_float32', tensor_type),
    ]
    schema = pa.schema(fields)  # type: ignore[arg-type]

    test_table = pa.Table.from_pydict(value_dict, schema=schema)
    parquet.write_table(test_table, str(output_path / 'test.parquet'))
    # read it back and verify the tables match
    read_table = parquet.read_table(str(output_path / 'test.parquet'))
    assert read_table.num_rows == test_table.num_rows
    assert set(read_table.column_names) == set(test_table.column_names)
    assert read_table.schema.equals(test_table.schema)
    assert read_table.equals(test_table)
    return str(output_path / 'test.parquet')


def assert_img_eq(img1: PIL.Image.Image, img2: PIL.Image.Image, context: str) -> None:
    assert img1.mode == img2.mode, context
    assert img1.size == img2.size, context
    diff = PIL.ImageChops.difference(img1, img2)  # type: ignore[attr-defined]
    assert diff.getbbox() is None, context


def reload_catalog() -> None:
    catalog.Catalog.clear()
    pxt.init()


clip_embed = clip.using(model_id='openai/clip-vit-base-patch32')
e5_embed = sentence_transformer.using(model_id='intfloat/e5-large-v2')


# Mock UDF for testing LLM tool invocations
@pxt.udf
def stock_price(ticker: str) -> float:
    """
    Get today's stock price for a given ticker symbol.

    Args:
        ticker - The ticker symbol of the stock to look up.
    """
    if ticker == 'NVDA':
        return 131.17
    else:
        # Return 0.0 instead of None, to distinguish between these two cases: the tool not being called, and the tool
        # being called on a symbol other than NVDA
        return 0.0


SAMPLE_IMAGE_URL = 'https://raw.githubusercontent.com/pixeltable/pixeltable/main/docs/resources/images/000000000009.jpg'


class ReloadTester:
    """Utility to verify that queries return identical results after a catalog reload"""

    df_info: list[tuple[dict[str, Any], DataFrameResultSet]]  # list of (df.as_dict(), df.collect())

    def __init__(self):
        self.df_info = []

    def clear(self) -> None:
        self.df_info = []

    def run_query(self, df: pxt.DataFrame) -> DataFrameResultSet:
        df_dict = df.as_dict()
        result_set = df.collect()
        self.df_info.append((df_dict, result_set))
        return result_set

    def run_reload_test(self, clear: bool = True) -> None:
        reload_catalog()
        for df_dict, result_set in self.df_info:
            df = pxt.DataFrame.from_dict(df_dict)
            new_result_set = df.collect()
            try:
                assert_resultset_eq(result_set, new_result_set, compare_col_names=True)
            except Exception as e:
                s = f'Reload test failed for query:\n{df}\n{e}'
                raise RuntimeError(s) from e
        if clear:
            self.clear()
