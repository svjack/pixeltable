from __future__ import annotations

import dataclasses
import logging
from typing import Optional, List, Dict, Any, Union, Tuple, Set, Callable
from uuid import UUID
import json

import sqlalchemy as sql

from .globals import is_valid_identifier, is_system_column_name
from .table import Table
from .column import Column
from .table_version import TableVersion
from ..env import Env
from ..metadata import schema
from pixeltable import exceptions as excs
import pixeltable.type_system as ts
import pixeltable.exprs as exprs


_logger = logging.getLogger('pixeltable')

class MutableTable(Table):
    """Base class for tables that allow mutations, ie, InsertableTable and View"""

    @dataclasses.dataclass
    class UpdateStatus:
        num_rows: int = 0
        # TODO: change to num_computed_columns (the number of computed slots isn't really meaningful to the user)
        num_computed_values: int = 0
        num_excs: int = 0
        updated_cols: List[str] = dataclasses.field(default_factory=list)
        cols_with_excs: List[str] = dataclasses.field(default_factory=list)

    def __init__(self, id: UUID, dir_id: UUID, tbl_version: TableVersion):
        super().__init__(id, dir_id, tbl_version.name, tbl_version)

    def move(self, new_name: str, new_dir_id: UUID) -> None:
        super().move(new_name, new_dir_id)
        with Env.get().engine.begin() as conn:
            stmt = sql.text((
                f"UPDATE {schema.Table.__table__} "
                f"SET {schema.Table.dir_id.name} = :new_dir_id, "
                f"    {schema.Table.md.name}['name'] = :new_name "
                f"WHERE {schema.Table.id.name} = :id"))
            conn.execute(stmt, {'new_dir_id': new_dir_id, 'new_name': json.dumps(new_name), 'id': self.id})

    def __setitem__(self, column_name: str, value: Union[ts.ColumnType, exprs.Expr, Callable, dict]) -> None:
        """Adds a column to the table
        Args:
            column_name: the name of the new column
            value: column type or value expression or column specification dictionary:
                column type: a Pixeltable column type (if the table already contains rows, it must be nullable)
                value expression: a Pixeltable expression that computes the column values
                column specification: a dictionary with possible keys 'type', 'value', 'stored', 'indexed'
        Examples:
            Add an int column with ``None`` values:

            >>> tbl['new_col'] = IntType(nullable=True)

            For a table with int column ``int_col``, add a column that is the factorial of ``int_col``. The names of
            the parameters of the Callable must correspond to existing column names (the column values are then passed
            as arguments to the Callable). In this case, the return type cannot be inferred and needs to be specified
            explicitly:

            >>> tbl['factorial'] = {'value': lambda int_col: math.factorial(int_col), 'type': IntType()}

            For a table with an image column ``frame``, add an image column ``rotated`` that rotates the image by
            90 degrees. In this case, the column type is inferred from the expression. Also, the column is not stored
            (by default, computed image columns are not stored but recomputed on demand):

            >>> tbl['rotated'] = tbl.frame.rotate(90)

            Do the same, but now the column is stored:

            >>> tbl['rotated'] = {'value': tbl.frame.rotate(90), 'stored': True}

            Add a resized version of the ``frame`` column and index it. The column does not need to be stored in order
            to be indexed:

            >>> tbl['small_frame'] = {'value': tbl.frame.resize([224, 224]), 'indexed': True}
        """
        if not isinstance(column_name, str):
            raise excs.Error(f'Column name must be a string, got {type(column_name)}')
        if not is_valid_identifier(column_name):
            raise excs.Error(f'Invalid column name: {column_name!r}')

        new_col = self._create_columns({column_name: value})[0]
        self._verify_column(new_col, self.column_names())
        return self.tbl_version.add_column(new_col)

    def add_column(
            self, *,
            type: Optional[ts.ColumnType] = None, stored: Optional[bool] = None, indexed: Optional[bool] = None,
            print_stats: bool = False, **kwargs: Any
    ) -> MutableTable.UpdateStatus:
        """Adds a column to the table.

        Args:
            kwargs: Exactly one keyword argument of the form ``column-name=type|value-expression``.
            type: The type of the column. Only valid and required if ``value-expression`` is a Callable.
            stored: Whether the column is materialized and stored or computed on demand. Only valid for image columns.
            indexed: Whether the column is indexed.
            print_stats: If ``True``, print execution metrics.

        Returns:
            execution status

        Raises:
            Error: If the column name is invalid or already exists.

        Examples:
            Add an int column with ``None`` values:

            >>> tbl.add_column(new_col=IntType())

            For a table with int column ``int_col``, add a column that is the factorial of ``int_col``. The names of
            the parameters of the Callable must correspond to existing column names (the column values are then passed
            as arguments to the Callable). In this case, the column type needs to be specified explicitly:

            >>> tbl.add_column(factorial=lambda int_col: math.factorial(int_col), type=IntType())

            For a table with an image column ``frame``, add an image column ``rotated`` that rotates the image by
            90 degrees. In this case, the column type is inferred from the expression. Also, the column is not stored
            (by default, computed image columns are not stored but recomputed on demand):

            >>> tbl.add_column(rotated=tbl.frame.rotate(90))

            Do the same, but now the column is stored:

            >>> tbl.add_column(rotated=tbl.frame.rotate(90), stored=True)

            Add a resized version of the ``frame`` column and index it. The column does not need to be stored in order
            to be indexed:

            >>> tbl.add_column(small_frame=tbl.frame.resize([224, 224]), indexed=True)
        """
        self._check_is_dropped()
        # verify kwargs and construct column schema dict
        if len(kwargs) != 1:
            raise excs.Error((
                f'add_column() requires exactly one keyword argument of the form "column-name=type|value-expression", '
                f'got {len(kwargs)} instead ({", ".join(list(kwargs.keys()))})'
            ))
        col_name, spec = next(iter(kwargs.items()))
        col_schema: Dict[str, Any] = {}
        if isinstance(spec, ts.ColumnType):
            if type is not None:
                raise excs.Error(f'add_column(): keyword argument "type" is redundant')
            col_schema['type'] = spec
        else:
            if isinstance(spec, exprs.Expr) and type is not None:
                raise excs.Error(f'add_column(): keyword argument "type" is redundant')
            col_schema['value'] = spec
        if type is not None:
            col_schema['type'] = type
        if stored is not None:
            col_schema['stored'] = stored
        if indexed is not None:
            col_schema['indexed'] = indexed

        new_col = self._create_columns({col_name: col_schema})[0]
        self._verify_column(new_col, self.column_names())
        return self.tbl_version.add_column(new_col, print_stats=print_stats)

    @classmethod
    def _validate_column_spec(cls, name: str, spec: Dict[str, Any]) -> None:
        """Check integrity of user-supplied Column spec

        We unfortunately can't use something like jsonschema for validation, because this isn't strictly a JSON schema
        (on account of containing Python Callables or Exprs).
        """
        assert isinstance(spec, dict)
        valid_keys = {'type', 'value', 'stored', 'indexed'}
        has_type = False
        for k in spec.keys():
            if k not in valid_keys:
                raise excs.Error(f'Column {name}: invalid key {k!r}')

        if 'type' in spec:
            has_type = True
            if not isinstance(spec['type'], ts.ColumnType):
                raise excs.Error(f'Column {name}: "type" must be a ColumnType, got {spec["type"]}')

        if 'value' in spec:
            value_spec = spec['value']
            value_expr = exprs.Expr.from_object(value_spec)
            if value_expr is None:
                # needs to be a Callable
                if not isinstance(value_spec, Callable):
                    raise excs.Error(
                        f'Column {name}: value needs to be either a Pixeltable expression or a Callable, '
                        f'but it is a {type(value_spec)}')
                if 'type' not in spec:
                    raise excs.Error(f'Column {name}: "type" is required if value is a Callable')
            else:
                has_type = True
                if 'type' in spec:
                    raise excs.Error(f'Column {name}: "type" is redundant if value is a Pixeltable expression')

        if 'stored' in spec and not isinstance(spec['stored'], bool):
            raise excs.Error(f'Column {name}: "stored" must be a bool, got {spec["stored"]}')
        if 'indexed' in spec and not isinstance(spec['indexed'], bool):
            raise excs.Error(f'Column {name}: "indexed" must be a bool, got {spec["indexed"]}')
        if not has_type:
            raise excs.Error(f'Column {name}: "type" is required')

    @classmethod
    def _create_columns(cls, schema: Dict[str, Any]) -> List[Column]:
        """Construct list of Columns, given schema"""
        columns: List[Column] = []
        for name, spec in schema.items():
            col_type: Optional[ts.ColumnType] = None
            value_expr: Optional[exprs.Expr] = None
            stored: Optional[bool] = None
            indexed: Optional[bool] = None
            primary_key: Optional[bool] = None

            if isinstance(spec, ts.ColumnType):
                col_type = spec
            elif isinstance(spec, exprs.Expr):
                value_expr = spec
            elif isinstance(spec, dict):
                cls._validate_column_spec(name, spec)
                col_type = spec.get('type')
                value_expr = spec.get('value')
                stored = spec.get('stored')
                indexed = spec.get('indexed')
                primary_key = spec.get('primary_key')

            column = Column(
                name, col_type=col_type, computed_with=value_expr, stored=stored, indexed=indexed,
                primary_key=primary_key)
            columns.append(column)
        return columns

    @classmethod
    def _verify_column(cls, col: Column, existing_column_names: Set[str]) -> None:
        """Check integrity of user-supplied Column and supply defaults"""
        if is_system_column_name(col.name):
            raise excs.Error(f'Column name {col.name} is reserved')
        if not is_valid_identifier(col.name):
            raise excs.Error(f"Invalid column name: '{col.name}'")
        if col.name in existing_column_names:
            raise excs.Error(f'Duplicate column name: {col.name}')
        if col.stored is False and not (col.is_computed and col.col_type.is_image_type()):
            raise excs.Error(f'Column {col.name}: stored={col.stored} only applies to computed image columns')
        if col.stored is False and not (col.col_type.is_image_type() and not col.has_window_fn_call()):
            raise excs.Error(
                f'Column {col.name}: stored={col.stored} is not valid for image columns computed with a streaming function')
        if col.stored is None:
            col.stored = not (col.is_computed and col.col_type.is_image_type() and not col.has_window_fn_call())

    @classmethod
    def _verify_schema(cls, schema: List[Column]) -> None:
        """Check integrity of user-supplied schema and set defaults"""
        column_names: Set[str] = set()
        for col in schema:
            cls._verify_column(col, column_names)
            column_names.add(col.name)

    def drop_column(self, name: str) -> None:
        """Drop a column from the table.

        Args:
            name: The name of the column to drop.

        Raises:
            Error: If the column does not exist or if it is referenced by a computed column.

        Examples:
            Drop column ``factorial``:

            >>> tbl.drop_column('factorial')
        """
        self._check_is_dropped()
        self.tbl_version.drop_column(name)

    def rename_column(self, old_name: str, new_name: str) -> None:
        """Rename a column.

        Args:
            old_name: The current name of the column.
            new_name: The new name of the column.

        Raises:
            Error: If the column does not exist or if the new name is invalid or already exists.

        Examples:
            >>> tbl.rename_column('factorial', 'fac')
        """
        self._check_is_dropped()
        self.tbl_version.rename_column(old_name, new_name)

    def update(
            self, value_spec: Dict[str, Union['pixeltable.exprs.Expr', Any]],
            where: Optional['pixeltable.exprs.Predicate'] = None, cascade: bool = True
    ) -> MutableTable.UpdateStatus:
        """Update rows in this table.
        Args:
            value_spec: a dict mapping column names to literal values or Pixeltable expressions.
            where: a Predicate to filter rows to update.
            cascade: if True, also update all computed columns that transitively depend on the updated columns.
        """
        from pixeltable import exprs
        update_targets: List[Tuple[Column, exprs.Expr]] = []
        for col_name, val in value_spec.items():
            if not isinstance(col_name, str):
                raise excs.Error(f'Update specification: dict key must be column name, got {col_name!r}')
            if col_name not in self.tbl_version.cols_by_name:
                raise excs.Error(f'Column {col_name} unknown')
            col = self.tbl_version.cols_by_name[col_name]
            if col.is_computed:
                raise excs.Error(f'Column {col_name} is computed and cannot be updated')
            if col.primary_key:
                raise excs.Error(f'Column {col_name} is a primary key column and cannot be updated')
            if col.col_type.is_media_type():
                raise excs.Error(f'Column {col_name} has type image/video/audio and cannot be updated')

            # make sure that the value is compatible with the column type
            # check if this is a literal
            try:
                value_expr = exprs.Literal(val, col_type=col.col_type)
            except TypeError:
                # it's not a literal, let's try to create an expr from it
                value_expr = exprs.Expr.from_object(val)
                if value_expr is None:
                    raise excs.Error(f'Column {col_name}: value {val!r} is not a recognized literal or expression')
                if not col.col_type.matches(value_expr.col_type):
                    raise excs.Error((
                        f'Type of value {val!r} ({value_expr.col_type}) is not compatible with the type of column '
                        f'{col_name} ({col.col_type})'
                    ))
            update_targets.append((col, value_expr))

        from pixeltable.plan import Planner
        if where is not None:
            if not isinstance(where, exprs.Predicate):
                raise excs.Error(f"'where' argument must be a Predicate, got {type(where)}")
            analysis_info = Planner.analyze(self.tbl_version, where)
            if analysis_info.similarity_clause is not None:
                raise excs.Error('nearest() cannot be used with update()')
            # for now we require that the updated rows can be identified via SQL, rather than via a Python filter
            if analysis_info.filter is not None:
                raise excs.Error(f'Filter {analysis_info.filter} not expressible in SQL')

        return self.tbl_version.update(update_targets, where, cascade)

    def revert(self) -> None:
        """Reverts the table to the previous version.

        .. warning::
            This operation is irreversible.
        """
        self._check_is_dropped()
        self.tbl_version.revert()

