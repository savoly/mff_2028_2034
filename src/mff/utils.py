from decimal import ROUND_HALF_UP, Decimal, getcontext
import pandas as pd
from sqlalchemy.engine import Engine


round_context = getcontext()
round_context.rounding = ROUND_HALF_UP


def c_round(x: float, digits: int, precision: int = 10) -> float:
    tmp = round(Decimal(x), precision)
    return float(tmp.__round__(digits))


def find_columns(
    col_names: list[str], engine: Engine, allowed_schemas: list[str] | None = None
) -> pd.DataFrame:
    cols = ", ".join([f"'%%{c}%%'" for c in col_names])
    arr = f"ARRAY[{cols}]"

    if allowed_schemas is None:
        sql = f"""
        SELECT table_schema, table_name, column_name, data_type
        FROM information_schema.columns
        WHERE column_name ILIKE ANY ({arr})
          AND table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY table_schema, table_name;"""

    else:
        if len(allowed_schemas) == 1:
            schemas = f"ARRAY['{allowed_schemas[0]}']"
        else:
            schemas_inner = ",".join([f"'{schema}'" for schema in allowed_schemas])
            schemas = f"ARRAY[{schemas_inner}]"
        sql = f"""
        SELECT table_schema, table_name, column_name, data_type
        FROM information_schema.columns
        WHERE column_name ILIKE ANY ({arr})
          AND table_schema ILIKE ANY ({schemas})
        ORDER BY table_schema, table_name;"""

    return pd.read_sql(sql, con=engine)
