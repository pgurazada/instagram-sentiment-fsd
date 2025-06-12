import sqlite3
import os
import pytest

DB_PATH = "interactions.db"

def get_logs(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT id, comment, sentiment, score, timestamp FROM interactions ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    return rows

def test_logs_exist():
    # Skip test if database does not exist
    if not os.path.exists(DB_PATH):
        pytest.skip("interactions.db does not exist")
    logs = get_logs()
    assert isinstance(logs, list)
    # If there are logs, check structure
    if logs:
        for row in logs:
            assert len(row) == 5
            assert isinstance(row[0], int)  # id
            assert isinstance(row[1], str)  # comment
            assert isinstance(row[2], str)  # sentiment
            assert isinstance(row[3], (int, float))  # score
            assert isinstance(row[4], str)  # timestamp