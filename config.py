from sqlalchemy import create_engine 
from urllib.parse import quote_plus

DB_USERNAME="your username"
DB_PASSWORD="your password"
DB_HOST="your host"
DB_NAME=" your db_name"

PASSWORD=quote_plus(DB_PASSWORD)
ENGINE = create_engine(
    f"mysql+pymysql://:{PASSWORD}/{DB_NAME}"
)

CSV_PATH=r"your file path"
TABLE_NAME="table name"