import os, psycopg
from dotenv import load_dotenv
load_dotenv()
url = os.environ["DATABASE_URL"]
print(url.split("@")[-1])
conn = psycopg.connect(url, connect_timeout=30)
print(conn.execute("select 1").fetchone())