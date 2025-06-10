import sqlite3

def read_logs(db_path="interactions.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT id, comment, sentiment, score, timestamp FROM interactions ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    if not rows:
        print("No logs found.")
        return
    for row in rows:
        print(f"ID: {row[0]}")
        print(f"Timestamp: {row[4]}")
        print(f"Comment: {row[1]}")
        print(f"Sentiment: {row[2]}")
        print(f"Score: {row[3]}")
        print("-" * 40)

if __name__ == "__main__":
    read_logs()