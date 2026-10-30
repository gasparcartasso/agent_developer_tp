from pathlib import Path
import pandas as pd
import sqlite3

# 1. Define paths
db_path = Path("app/data/sql/member_info.db")
csv_path = "data/member_info.csv"

# 2. Ensure the parent directories exist (creates them if missing)
db_path.parent.mkdir(parents=True, exist_ok=True)

# 3. Load CSV and connect
df = pd.read_csv(csv_path)
conn = sqlite3.connect(db_path)

# 4. Write to SQL
df.to_sql(
    "member_info",
    conn,
    dtype={"member_number": "INTEGER", "product": "TEXT"},
    if_exists="replace",
    index=False,
)

conn.close()
print(f"Successfully created database at {db_path}")