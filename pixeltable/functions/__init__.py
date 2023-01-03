from typing import Callable, List, Optional
import inspect

from pixeltable.type_system import StringType, IntType, JsonType, ColumnType
from pixeltable.function import Function
from pixeltable import catalog
from pixeltable import exprs
import pixeltable.exceptions as exc


def udf_call(eval_fn: Callable, return_type: ColumnType, tbl: Optional[catalog.Table]) -> exprs.FunctionCall:
    """
    Interprets eval_fn's parameters to be references to columns in 'tbl' and construct ColumnRefs as args.
    """
    params = inspect.signature(eval_fn).parameters
    if len(params) > 0 and tbl is None:
        raise exc.OperationalError(f'udf_call() is missing tbl parameter')
    args: List[exprs.ColumnRef] = []
    for param_name in params:
        if param_name not in tbl.cols_by_name:
            raise exc.OperationalError(
                (f'udf_call(): lambda argument names need to be valid column names in table {tbl.name}: '
                 f'column {param_name} unknown'))
        args.append(exprs.ColumnRef(tbl.cols_by_name[param_name]))
    fn = Function(return_type, [arg.col_type for arg in args], eval_fn=eval_fn)
    return exprs.FunctionCall(fn, args)

def cast(expr: exprs.Expr, target_type: ColumnType) -> exprs.Expr:
    expr.col_type = target_type
    return expr

dict_map = Function(IntType(), [StringType(), JsonType()], eval_fn=lambda s, d: d[s])

__all__ = [
    udf_call,
    cast,
    dict_map
]
