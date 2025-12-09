import sqlite3

conn = sqlite3.connect('job_predictions.db')

conn.execute('''
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_description TEXT,
    prediction TEXT,
    confidence REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
''')

conn.close()
print("✅ Database and table created successfully!")
