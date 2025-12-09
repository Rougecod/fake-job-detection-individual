# app.py
from flask import Flask, render_template, request, redirect, session, url_for
import joblib
import sqlite3
import re
from datetime import datetime
import json
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# --- Paths (adjust if your files are in different locations) ---
MODEL_PATH = 'fake_job_model.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'
DB_PATH = 'job_predictions.db'

# --- Load model/vectorizer (wrap in try so app still starts even if model mismatch) ---
model = None
vectorizer = None
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"[WARN] could not load model from {MODEL_PATH}: {e}")

try:
    vectorizer = joblib.load(VECTORIZER_PATH)
except Exception as e:
    print(f"[WARN] could not load vectorizer from {VECTORIZER_PATH}: {e}")


# ---------- Database helpers ----------
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_description TEXT,
            prediction TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS admin (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        );
    ''')
    # create default admin if not exists
    existing = conn.execute("SELECT * FROM admin WHERE username = 'admin'").fetchone()
    if not existing:
        conn.execute("INSERT INTO admin (username, password) VALUES (?, ?)", ('admin', 'admin123'))
        print("Created default admin -> username: 'admin' password: 'admin123'")
    conn.commit()
    conn.close()


init_db()


# ---------- ADMIN LOGIN ----------
@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        conn = get_db_connection()
        cursor = conn.execute("SELECT * FROM admin WHERE username=? AND password=?", (username, password))
        admin = cursor.fetchone()
        conn.close()

        if admin:
            session['admin_logged_in'] = True
            session['admin_username'] = username
            return redirect(url_for('admin_dashboard'))
        else:
            return render_template('Admin_login.html', error="Invalid username or password")

    return render_template('Admin_login.html')


# ---------- ADMIN DASHBOARD ----------
@app.route('/admin_dashboard')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))

    conn = get_db_connection()

    # counts
    fake_count = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'Fake Job'").fetchone()[0]
    real_count = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'Real Job'").fetchone()[0]
    total = fake_count + real_count

    # compute a simple "accuracy" metric (since we don't have ground-truth labels in this example,
    # we'll show % of Real Jobs as a proxy). Adjust logic if you have a different definition.
    accuracy = round((real_count / total) * 100, 1) if total > 0 else 0.0

    # daily counts for chart (string dates)
    daily_data = conn.execute("""
        SELECT DATE(timestamp) as day, COUNT(*) as cnt
        FROM predictions
        GROUP BY DATE(timestamp)
        ORDER BY DATE(timestamp)
    """).fetchall()

    dates = [row['day'] for row in daily_data]
    counts = [row['cnt'] for row in daily_data]

    conn.close()

    # Convert lists to JSON strings so Chart.js gets proper arrays in template
    return render_template(
        'Admin_dashboard.html',
        fake=fake_count,
        real=real_count,
        total=total,
        accuracy=accuracy,
        dates=json.dumps(dates),
        counts=json.dumps(counts)
    )


# ---------- INDEX PAGE (prediction form) ----------
@app.route('/index_page')
def index_page():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    return render_template('index.html')


# ---------- LOGOUT ----------
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('admin_login'))


# ---------- PREDICT ----------
@app.route('/predict', methods=['POST'])
def predict():
    job_desc = request.form.get('job_description', '').strip()

    if not job_desc:
        return render_template('index.html', error="Please enter a job description.")
    if not re.search(r'[A-Za-z]', job_desc):
        return render_template('index.html', error="Please enter valid text.")

    if vectorizer is None or model is None:
        return render_template('index.html', error="Model or vectorizer not loaded on the server.")

    try:
        X_input = vectorizer.transform([job_desc])
        pred_raw = model.predict(X_input)[0]
    except Exception as e:
        return render_template('index.html', error=f"Prediction failed: {e}")

    # try to get probability for confidence - some models may not support predict_proba
    confidence = None
    try:
        proba = model.predict_proba(X_input)[0]
        # If model classes are [0,1] and 1 means Fake Job (depends on how you trained)
        # We'll map based on predicted class.
        if pred_raw == 1:
            confidence = round(proba[1] * 100, 2)
        else:
            confidence = round(proba[0] * 100, 2)
    except Exception:
        # fallback: no predict_proba, mark confidence as 0
        confidence = 0.0

    label = "Fake Job" if pred_raw == 1 else "Real Job"

    # store prediction
    conn = get_db_connection()
    conn.execute(
        'INSERT INTO predictions (job_description, prediction, confidence) VALUES (?, ?, ?)',
        (job_desc, label, confidence)
    )
    conn.commit()
    conn.close()

    return render_template('result.html', label=label, confidence=confidence, description=job_desc)


# ---------- HISTORY ----------
@app.route('/history')
def history():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))

    conn = get_db_connection()
    cursor = conn.execute('SELECT job_description, prediction, confidence, timestamp FROM predictions ORDER BY id DESC')
    records = cursor.fetchall()
    conn.close()
    return render_template('history.html', records=records)


# ---------- Home redirect ----------
@app.route('/')
def home():
    return redirect(url_for('admin_login'))


if __name__ == '__main__':
    # Ensure working directory is the project root (so Flask finds templates folder)
    # If you run this from a different directory, change cwd accordingly:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    app.run(debug=True)
