from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    Integer,
    Float,
    String,
    inspect,
    text,
)
from sqlalchemy.exc import OperationalError
from sqlalchemy.sql import insert
import os

# DB path (relative to project root)
DB_FILE = os.path.join("app", "database", "results.db")
DB_URL = f"sqlite:///{DB_FILE}"

# ensure directory exists
os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)

# engine with check_same_thread False for web apps
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})

metadata = MetaData()

# Define expected table schemas explicitly
credit_results = Table(
    "credit_results",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("age", Float, nullable=True),
    Column("rf_prob", Float, nullable=True),
    Column("gb_prob", Float, nullable=True),
)

# map of table name -> Table object for easy lookup
_TABLES = {"credit_results": credit_results}


def _sql_type_name(col):
    """Return a simple SQL type name for ALTER TABLE statements."""
    typ = type(col.type)
    if typ is Integer:
        return "INTEGER"
    if typ is Float:
        return "FLOAT"
    return "TEXT"


def _ensure_columns(table_name: str):
    """Check existing columns and add any missing ones via ALTER TABLE."""
    inspector = inspect(engine)
    if not inspector.has_table(table_name):
        return

    existing = {c["name"] for c in inspector.get_columns(table_name)}
    target_table = _TABLES.get(table_name)
    if target_table is None:
        return

    for col in target_table.columns:
        if col.name not in existing:
            # SQLite supports ADD COLUMN with simple types
            col_type = _sql_type_name(col)
            sql = text(f"ALTER TABLE {table_name} ADD COLUMN {col.name} {col_type}")
            with engine.begin() as conn:
                conn.execute(sql)


def init_db():
    """Create expected tables and migrate missing columns.

    Call this once at application startup.
    """
    # create any missing tables
    metadata.create_all(engine)

    # ensure columns exist (lightweight migration)
    for tname in _TABLES.keys():
        try:
            _ensure_columns(tname)
        except OperationalError:
            # if any SQL error occurs during migration, skip to avoid crashing startup
            continue


def save_result(data: dict, table: str):
    """Insert a result row into a known table using SQLAlchemy insert().

    Only `credit_results` and `fraud_results` are supported by design.
    """
    if table not in _TABLES:
        raise ValueError(f"Unknown table: {table}")

    tbl = _TABLES[table]
    # insert using begin() for transactional safety
    with engine.begin() as conn:
        conn.execute(insert(tbl).values(**data))
