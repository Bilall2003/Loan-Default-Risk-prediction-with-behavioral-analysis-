from sqlalchemy import create_engine 
from urllib.parse import quote_plus

DB_USERNAME="root"
DB_PASSWORD="Bilalahmed@1234"
DB_HOST="localhost"
DB_NAME="practice2_seperatefile_method"

PASSWORD=quote_plus(DB_PASSWORD)
ENGINE = create_engine(
    f"mysql+pymysql://app_user:{PASSWORD}@localhost/{DB_NAME}"
)

CSV_PATH=r"D:\Bilal folder\internship\task4\credit_risk_dataset.csv"
TABLE_NAME="data_info"