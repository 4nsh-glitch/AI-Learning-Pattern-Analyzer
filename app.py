from flask import Flask, request, render_template_string, redirect, url_for, send_file
import sqlite3
import os
from datetime import datetime
import pandas as pd
import io

try:
    import joblib
except ImportError:
    joblib = None


# =========================================================
# APP CONFIG
# =========================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "student_history.db")
MODEL_PATH = os.path.join(BASE_DIR, "learning_pattern_model.pkl")


# =========================================================
# EXAM + SUBJECT DATA
# =========================================================

EXAM_SUBJECTS = {
    "SSC CGL": ["Maths", "Reasoning", "English", "GK"],
    "SSC CHSL": ["Maths", "Reasoning", "English", "GK"],
    "SSC GD": ["GK", "General Science", "Reasoning", "Maths"],
    "SSC CPO": ["Maths", "Reasoning", "English", "GK"],
    "Railway NTPC": ["Maths", "Reasoning", "General Science", "GK"],
    "Railway Group D": ["Maths", "Reasoning", "General Science", "GK"],
}


SUBJECT_TOPICS = {
    "Maths": [
        "Percentage",
        "Ratio",
        "Profit & Loss",
        "Time & Work",
        "Mensuration"
    ],
    "Reasoning": [
        "Analogy",
        "Series",
        "Coding-Decoding",
        "Blood Relation",
        "Syllogism"
    ],
    "English": [
        "Grammar",
        "Vocabulary",
        "Error Detection",
        "Cloze Test",
        "Reading Comprehension"
    ],
    "GK": [
        "History",
        "Geography",
        "Polity",
        "Economy",
        "Current Affairs"
    ],
    "General Science": [
        "Physics",
        "Chemistry",
        "Biology",
        "Environment",
        "Everyday Science"
    ]
}


# =========================================================
# DATABASE
# =========================================================

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT,
            exam TEXT,
            subject TEXT,
            study_hours REAL,
            mock_score REAL,
            questions_attempted INTEGER,
            correct_answers INTEGER,
            wrong_answers INTEGER,
            accuracy REAL,
            prediction REAL,
            level TEXT,
            subject_analysis TEXT,
            date_time TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT,
            exam TEXT,
            subject TEXT,
            topic TEXT,
            score REAL,
            date_time TEXT
        )
    """)

    conn.commit()
    conn.close()


init_db()


# =========================================================
# LOAD MODEL
# =========================================================

model = None

if joblib is not None and os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        model = None


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_level(prediction):
    if prediction >= 80:
        return "Excellent"
    elif prediction >= 60:
        return "Good"
    elif prediction >= 40:
        return "Needs Improvement"
    return "Low"


def subject_analysis(accuracy):
    if accuracy >= 80:
        return "Strong"
    elif accuracy >= 60:
        return "Moderate"
    return "Weak"


def get_topic_status(score):
    if score >= 80:
        return "Strong"
    elif score >= 60:
        return "Moderate"
    return "Weak"


def predict_performance(
    study_hours,
    mock_score,
    questions_attempted,
    correct_answers,
    wrong_answers,
    accuracy
):
    """
    Uses the existing Random Forest model.

    The DataFrame keeps the same feature names that were used
    while training the model.
    """

    features = pd.DataFrame([{
        "study_hours": study_hours,
        "mock_score": mock_score,
        "questions_attempted": questions_attempted,
        "correct_answers": correct_answers,
        "wrong_answers": wrong_answers,
        "accuracy": accuracy
    }])

    if model is not None:
        try:
            prediction = float(model.predict(features)[0])
        except Exception:
            prediction = calculate_fallback_prediction(
                study_hours,
                mock_score,
                accuracy
            )
    else:
        prediction = calculate_fallback_prediction(
            study_hours,
            mock_score,
            accuracy
        )

    prediction = max(0, min(100, prediction))

    return round(prediction, 2)


def calculate_fallback_prediction(study_hours, mock_score, accuracy):
    """
    Backup calculation only used if model is unavailable.
    """

    study_component = min(study_hours / 6, 1) * 20
    mock_component = mock_score * 0.40
    accuracy_component = accuracy * 0.40

    result = study_component + mock_component + accuracy_component

    return result


def generate_recommendations(
    study_hours,
    mock_score,
    questions_attempted,
    correct_answers,
    wrong_answers,
    accuracy
):
    recommendations = []

    if study_hours < 3:
        recommendations.append(
            "Increase daily study time to at least 3-4 hours."
        )

    if accuracy < 60:
        recommendations.append(
            "Improve accuracy by practicing questions slowly and carefully."
        )

    if wrong_answers > 15:
        recommendations.append(
            "Reduce wrong attempts and focus on question selection."
        )

    if mock_score < 60:
        recommendations.append(
            "Take regular mock tests and revise weak topics."
        )

    if questions_attempted > 0:
        if correct_answers < questions_attempted * 0.70:
            recommendations.append(
                "Practice more PYQs for this subject."
            )

    if not recommendations:
        recommendations.append(
            "Your preparation is going well. Continue regular mock tests and revision."
        )

    return recommendations


def detect_weak_area(
    study_hours,
    mock_score,
    accuracy,
    subject
):
    if accuracy < 60:
        return f"{subject} accuracy"

    if mock_score < 60:
        return "Mock test performance"

    if study_hours < 3:
        return "Daily study time"

    return "No major weakness"


def create_study_plan(
    study_hours,
    accuracy,
    mock_score
):
    if accuracy < 60:
        concept_time = 90
        pyq_time = 60
    else:
        concept_time = 60
        pyq_time = 45

    if mock_score < 60:
        mock_time = 45
    else:
        mock_time = 30

    plan = [
        {
            "task": "Concept Study",
            "time": f"{concept_time} min",
            "description": "Weak topics ke concepts revise karo."
        },
        {
            "task": "PYQ Practice",
            "time": f"{pyq_time} min",
            "description": "Previous Year Questions solve karo."
        },
        {
            "task": "Mock Test",
            "time": f"{mock_time} min",
            "description": "Timed mock test attempt karo."
        },
        {
            "task": "Mistake Analysis",
            "time": "20 min",
            "description": "Wrong questions ka reason check karo."
        },
        {
            "task": "Revision",
            "time": "20 min",
            "description": "Aaj padhe topics ka quick revision karo."
        }
    ]

    return plan


def get_badges(
    accuracy,
    prediction,
    questions_attempted,
    mock_score,
    study_hours
):
    badges = []

    if accuracy >= 80:
        badges.append(("🎯", "80%+ Accuracy"))

    if prediction >= 80:
        badges.append(("⭐", "Excellent Prediction"))

    if questions_attempted >= 100:
        badges.append(("🔥", "100+ Questions"))

    if mock_score >= 80:
        badges.append(("🏆", "80+ Mock Score"))

    if study_hours >= 4:
        badges.append(("📚", "4+ Hours Study"))

    if not badges:
        badges.append(("🚀", "Keep Improving"))

    return badges


def get_dashboard_data():
    conn = get_db()

    total_tests = conn.execute(
        "SELECT COUNT(*) AS count FROM results"
    ).fetchone()["count"]

    avg_accuracy_row = conn.execute(
        "SELECT AVG(accuracy) AS value FROM results"
    ).fetchone()

    avg_prediction_row = conn.execute(
        "SELECT AVG(prediction) AS value FROM results"
    ).fetchone()

    avg_accuracy = avg_accuracy_row["value"] or 0
    avg_prediction = avg_prediction_row["value"] or 0

    best_subject_row = conn.execute("""
        SELECT subject, AVG(accuracy) AS avg_accuracy
        FROM results
        GROUP BY subject
        ORDER BY avg_accuracy DESC
        LIMIT 1
    """).fetchone()

    best_subject = (
        best_subject_row["subject"]
        if best_subject_row
        else "N/A"
    )

    weak_subject_row = conn.execute("""
        SELECT subject, AVG(accuracy) AS avg_accuracy
        FROM results
        GROUP BY subject
        ORDER BY avg_accuracy ASC
        LIMIT 1
    """).fetchone()

    weak_subject = (
        weak_subject_row["subject"]
        if weak_subject_row
        else "N/A"
    )

    conn.close()

    return {
        "total_tests": total_tests,
        "avg_accuracy": round(avg_accuracy, 2),
        "avg_prediction": round(avg_prediction, 2),
        "best_subject": best_subject,
        "weak_subject": weak_subject
    }


def get_subject_progress():
    conn = get_db()

    rows = conn.execute("""
        SELECT
            subject,
            COUNT(*) AS tests,
            AVG(accuracy) AS avg_accuracy,
            AVG(prediction) AS avg_prediction
        FROM results
        GROUP BY subject
        ORDER BY avg_accuracy DESC
    """).fetchall()

    conn.close()

    return rows


def get_history():
    conn = get_db()

    rows = conn.execute("""
        SELECT *
        FROM results
        ORDER BY id DESC
    """).fetchall()

    conn.close()

    return rows


def get_topic_history():
    conn = get_db()

    rows = conn.execute("""
        SELECT *
        FROM topics
        ORDER BY id DESC
    """).fetchall()

    conn.close()

    return rows


def get_topic_latest(student_name, subject, topic):
    conn = get_db()

    row = conn.execute("""
        SELECT *
        FROM topics
        WHERE student_name = ?
        AND subject = ?
        AND topic = ?
        ORDER BY id DESC
        LIMIT 1
    """, (student_name, subject, topic)).fetchone()

    conn.close()

    return row


# =========================================================
# MAIN PAGE HTML
# =========================================================

HTML = r"""
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>AI Government Exam Performance Analyzer</title>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>

* {
    box-sizing: border-box;
}

:root {
    --primary: #2563eb;
    --primary-dark: #1d4ed8;
    --bg: #f1f5f9;
    --card: #ffffff;
    --text: #0f172a;
    --muted: #64748b;
    --border: #e2e8f0;
    --success: #16a34a;
    --danger: #dc2626;
    --warning: #d97706;
}

body {
    margin: 0;
    font-family: Arial, Helvetica, sans-serif;
    background: var(--bg);
    color: var(--text);
}

body.dark {
    --bg: #0f172a;
    --card: #1e293b;
    --text: #f8fafc;
    --muted: #cbd5e1;
    --border: #334155;
}

body.blue {
    --primary: #0284c7;
    --primary-dark: #0369a1;
}

body.green {
    --primary: #16a34a;
    --primary-dark: #15803d;
}

.container {
    width: 94%;
    max-width: 1400px;
    margin: auto;
}

.header {
    background: linear-gradient(
        135deg,
        var(--primary),
        var(--primary-dark)
    );
    color: white;
    padding: 24px 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15);
}

.header-content {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 20px;
}

.header h1 {
    margin: 0;
    font-size: 28px;
}

.header p {
    margin: 7px 0 0;
    opacity: 0.9;
}

.settings {
    background: rgba(255,255,255,0.15);
    padding: 12px;
    border-radius: 12px;
}

.settings select {
    padding: 7px;
    border-radius: 6px;
    border: none;
    margin: 3px;
}

.nav {
    background: var(--card);
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    z-index: 10;
}

.nav-inner {
    display: flex;
    gap: 8px;
    padding: 10px 0;
    overflow-x: auto;
}

.nav a {
    text-decoration: none;
    color: var(--text);
    padding: 9px 14px;
    border-radius: 8px;
    white-space: nowrap;
}

.nav a:hover {
    background: var(--primary);
    color: white;
}

.section {
    margin: 28px 0;
}

.section-title {
    margin-bottom: 16px;
}

.section-title h2 {
    margin: 0;
}

.section-title p {
    margin: 5px 0;
    color: var(--muted);
}

.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 22px;
    box-shadow: 0 5px 18px rgba(0,0,0,0.05);
}

.grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
}

.grid-2 {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 18px;
}

.stat-card {
    text-align: center;
}

.stat-icon {
    font-size: 28px;
}

.stat-value {
    font-size: 28px;
    font-weight: bold;
    margin-top: 8px;
}

.stat-label {
    color: var(--muted);
    margin-top: 4px;
}

form {
    display: grid;
    gap: 14px;
}

.form-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 14px;
}

label {
    font-weight: bold;
    display: block;
    margin-bottom: 6px;
}

input,
select {
    width: 100%;
    padding: 11px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--card);
    color: var(--text);
    font-size: 15px;
}

button,
.btn {
    display: inline-block;
    border: none;
    padding: 11px 17px;
    border-radius: 9px;
    background: var(--primary);
    color: white;
    cursor: pointer;
    text-decoration: none;
    font-size: 15px;
}

button:hover,
.btn:hover {
    background: var(--primary-dark);
}

.btn-danger {
    background: var(--danger);
}

.btn-success {
    background: var(--success);
}

.result-box {
    margin-top: 20px;
    padding: 20px;
    border-radius: 14px;
    background: rgba(37,99,235,0.08);
    border: 1px solid var(--primary);
}

.prediction {
    font-size: 45px;
    font-weight: bold;
    color: var(--primary);
}

.level {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 20px;
    background: var(--primary);
    color: white;
    font-weight: bold;
}

.progress {
    height: 12px;
    background: var(--border);
    border-radius: 20px;
    overflow: hidden;
    margin-top: 8px;
}

.progress-bar {
    height: 100%;
    background: var(--primary);
}

.recommendation {
    padding: 12px;
    margin: 8px 0;
    background: rgba(37,99,235,0.08);
    border-left: 4px solid var(--primary);
    border-radius: 6px;
}

.weak {
    color: var(--danger);
    font-weight: bold;
}

.strong {
    color: var(--success);
    font-weight: bold;
}

.moderate {
    color: var(--warning);
    font-weight: bold;
}

table {
    width: 100%;
    border-collapse: collapse;
}

th,
td {
    border-bottom: 1px solid var(--border);
    padding: 11px;
    text-align: left;
}

th {
    background: rgba(37,99,235,0.08);
}

.table-wrapper {
    overflow-x: auto;
}

.badges {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
}

.badge {
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    background: rgba(37,99,235,0.08);
    border: 1px solid var(--border);
}

.badge-icon {
    font-size: 30px;
}

.topic-status {
    padding: 5px 10px;
    border-radius: 15px;
    font-size: 13px;
    font-weight: bold;
}

.status-strong {
    background: #dcfce7;
    color: #166534;
}

.status-moderate {
    background: #fef3c7;
    color: #92400e;
}

.status-weak {
    background: #fee2e2;
    color: #991b1b;
}

.chart-container {
    position: relative;
    height: 330px;
}

.footer {
    text-align: center;
    color: var(--muted);
    padding: 30px 0;
}

.small {
    color: var(--muted);
    font-size: 13px;
}

.compact .card {
    padding: 14px;
}

.compact .section {
    margin: 18px 0;
}

.large-font {
    font-size: 18px;
}

.small-font {
    font-size: 14px;
}

@media (max-width: 1000px) {
    .grid {
        grid-template-columns: repeat(2, 1fr);
    }

    .header-content {
        flex-direction: column;
        align-items: flex-start;
    }
}

@media (max-width: 700px) {
    .grid,
    .grid-2,
    .form-grid,
    .badges {
        grid-template-columns: 1fr;
    }

    .header h1 {
        font-size: 22px;
    }
}

</style>

</head>

<body>

<header class="header">

<div class="container header-content">

<div>

<h1>🤖 AI Government Exam Performance Analyzer</h1>

<p>
AI-Based Learning Pattern Analyzer for Government Exam Preparation
</p>

</div>

<div class="settings">

<select id="theme">
<option value="light">Light</option>
<option value="dark">Dark</option>
<option value="blue">Blue</option>
<option value="green">Green</option>
</select>

<select id="fontSize">
<option value="normal">Normal</option>
<option value="small">Small</option>
<option value="large">Large</option>
</select>

<select id="layout">
<option value="comfortable">Comfortable</option>
<option value="compact">Compact</option>
</select>

</div>

</div>

</header>


<nav class="nav">

<div class="container nav-inner">

<a href="#analyze">Analyze</a>
<a href="#dashboard">Dashboard</a>
<a href="#subjects">Subjects</a>
<a href="#topics">Topics</a>
<a href="#history">History</a>

</div>

</nav>


<main class="container">


<!-- =====================================================
     ANALYZER
====================================================== -->

<section class="section" id="analyze">

<div class="section-title">

<h2>📊 Analyze Exam Performance</h2>

<p>
Student performance ko AI model se analyze karo.
</p>

</div>


<div class="card">

<form method="POST" action="/">

<div class="form-grid">

<div>

<label>Student Name</label>

<input
type="text"
name="student_name"
placeholder="Enter student name"
value="{{ form_data.student_name if form_data else '' }}"
required>

</div>


<div>

<label>Government Exam</label>

<select name="exam" id="exam" required>

<option value="">Select Exam</option>

{% for exam_name in exams %}

<option
value="{{ exam_name }}"
{% if form_data and form_data.exam == exam_name %}
selected
{% endif %}
>
{{ exam_name }}
</option>

{% endfor %}

</select>

</div>


<div>

<label>Subject</label>

<select name="subject" id="subject" required>

<option value="">Select Subject</option>

{% if form_data and form_data.subject %}

<option selected value="{{ form_data.subject }}">
{{ form_data.subject }}
</option>

{% endif %}

</select>

</div>


<div>

<label>Study Hours per Day</label>

<input
type="number"
step="0.1"
min="0"
name="study_hours"
value="{{ form_data.study_hours if form_data else '' }}"
required>

</div>


<div>

<label>Mock Test Score</label>

<input
type="number"
step="0.1"
min="0"
max="100"
name="mock_score"
value="{{ form_data.mock_score if form_data else '' }}"
required>

</div>


<div>

<label>Questions Attempted</label>

<input
type="number"
min="0"
name="questions_attempted"
id="questions_attempted"
value="{{ form_data.questions_attempted if form_data else '' }}"
required>

</div>


<div>

<label>Correct Answers</label>

<input
type="number"
min="0"
name="correct_answers"
id="correct_answers"
value="{{ form_data.correct_answers if form_data else '' }}"
required>

</div>


<div>

<label>Wrong Answers</label>

<input
type="number"
name="wrong_answers"
id="wrong_answers"
value="{{ form_data.wrong_answers if form_data else '0' }}"
readonly>

</div>


<div>

<label>Accuracy</label>

<input
type="number"
step="0.01"
name="accuracy"
id="accuracy"
value="{{ form_data.accuracy if form_data else '0' }}"
readonly>

</div>

</div>


<button type="submit">
🤖 Analyze Exam Performance
</button>

</form>

</div>


{% if result %}

<div class="result-box">

<div class="grid">

<div class="card stat-card">

<div class="stat-icon">🤖</div>

<div class="stat-value">
{{ result.prediction }}%
</div>

<div class="stat-label">
AI Prediction
</div>

</div>


<div class="card stat-card">

<div class="stat-icon">📈</div>

<div class="stat-value">
{{ result.accuracy }}%
</div>

<div class="stat-label">
Accuracy
</div>

</div>


<div class="card stat-card">

<div class="stat-icon">📝</div>

<div class="stat-value">
{{ result.mock_score }}%
</div>

<div class="stat-label">
Mock Score
</div>

</div>


<div class="card stat-card">

<div class="stat-icon">🏅</div>

<div class="stat-value">

<span class="level">
{{ result.level }}
</span>

</div>

<div class="stat-label">
Performance Level
</div>

</div>

</div>


<div class="grid-2" style="margin-top:20px;">


<div class="card">

<h3>🎯 Subject Analysis</h3>

<p>

Subject:
<strong>{{ result.subject }}</strong>

</p>

<p>

Status:

{% if result.subject_analysis == "Strong" %}

<span class="strong">
🟢 Strong
</span>

{% elif result.subject_analysis == "Moderate" %}

<span class="moderate">
🟡 Moderate
</span>

{% else %}

<span class="weak">
🔴 Weak
</span>

{% endif %}

</p>

<div class="progress">

<div
class="progress-bar"
style="width: {{ result.accuracy }}%">
</div>

</div>

</div>


<div class="card">

<h3>⚠️ Weak Area</h3>

<p style="font-size:20px;">

<strong>
{{ result.weak_area }}
</strong>

</p>

</div>


</div>


<div class="grid-2" style="margin-top:20px;">


<div class="card">

<h3>💡 Smart Recommendations</h3>

{% for recommendation in result.recommendations %}

<div class="recommendation">
💡 {{ recommendation }}
</div>

{% endfor %}

</div>


<div class="card">

<h3>📅 Personalized Study Plan</h3>

{% for item in result.study_plan %}

<div class="recommendation">

<strong>
{{ item.task }}
</strong>

<br>

<span class="small">
{{ item.time }} — {{ item.description }}
</span>

</div>

{% endfor %}

</div>


</div>


<div class="card" style="margin-top:20px;">

<h3>🏆 Achievements</h3>

<div class="badges">

{% for icon, badge in result.badges %}

<div class="badge">

<div class="badge-icon">
{{ icon }}
</div>

<strong>
{{ badge }}
</strong>

</div>

{% endfor %}

</div>

</div>


<div class="card" style="margin-top:20px;">

<h3>📊 Current Performance Chart</h3>

<div class="chart-container">

<canvas id="currentChart"></canvas>

</div>

</div>


<div style="margin-top:20px;">

<a
class="btn btn-success"
href="/download_report/{{ result.id }}"
>
📄 Download Report
</a>

</div>


</div>

{% endif %}

</section>


<!-- =====================================================
     DASHBOARD
====================================================== -->

<section class="section" id="dashboard">

<div class="section-title">

<h2>📊 Student Dashboard</h2>

<p>
Overall preparation performance.
</p>

</div>


<div class="grid">

<div class="card stat-card">

<div class="stat-icon">📝</div>

<div class="stat-value">
{{ dashboard.total_tests }}
</div>

<div class="stat-label">
Total Tests
</div>

</div>


<div class="card stat-card">

<div class="stat-icon">🎯</div>

<div class="stat-value">
{{ dashboard.avg_accuracy }}%
</div>

<div class="stat-label">
Average Accuracy
</div>

</div>


<div class="card stat-card">

<div class="stat-icon">🤖</div>

<div class="stat-value">
{{ dashboard.avg_prediction }}%
</div>

<div class="stat-label">
Average AI Prediction
</div>

</div>


<div class="card stat-card">

<div class="stat-icon">⭐</div>

<div class="stat-value">
{{ dashboard.best_subject }}
</div>

<div class="stat-label">
Best Subject
</div>

</div>

</div>


<div class="grid-2" style="margin-top:18px;">

<div class="card">

<h3>🏆 Strongest Subject</h3>

<p style="font-size:24px;">
{{ dashboard.best_subject }}
</p>

</div>


<div class="card">

<h3>⚠️ Current Weakest Subject</h3>

<p style="font-size:24px;">
{{ dashboard.weak_subject }}
</p>

</div>

</div>

</section>


<!-- =====================================================
     SUBJECT PROGRESS
====================================================== -->

<section class="section" id="subjects">

<div class="section-title">

<h2>📚 Subject-wise Progress</h2>

<p>
Har subject ki average performance.
</p>

</div>


<div class="card">

{% if subjects %}

{% for row in subjects %}

<div style="margin-bottom:20px;">

<div style="
display:flex;
justify-content:space-between;
">

<strong>
{{ row.subject }}
</strong>

<span>
{{ "%.2f"|format(row.avg_accuracy or 0) }}%
</span>

</div>

<div class="progress">

<div
class="progress-bar"
style="width: {{ row.avg_accuracy or 0 }}%">
</div>

</div>

<p class="small">

Tests: {{ row.tests }}

|

Average Prediction:
{{ "%.2f"|format(row.avg_prediction or 0) }}%

</p>

</div>

{% endfor %}

{% else %}

<p class="small">
Abhi subject data available nahi hai.
Pehle performance analyze karo.
</p>

{% endif %}

</div>

</section>


<!-- =====================================================
     TOPIC TRACKER
====================================================== -->

<section class="section" id="topics">

<div class="section-title">

<h2>🎯 Topic-wise Weak Area Tracker</h2>

<p>
Specific topic ki preparation track karo.
</p>

</div>


<div class="grid-2">


<div class="card">

<form method="POST" action="/save_topic">

<div>

<label>Student Name</label>

<input
type="text"
name="student_name"
placeholder="Student name"
required>

</div>


<div>

<label>Exam</label>

<select name="exam" required>

<option value="">
Select Exam
</option>

{% for exam_name in exams %}

<option value="{{ exam_name }}">
{{ exam_name }}
</option>

{% endfor %}

</select>

</div>


<div>

<label>Subject</label>

<select name="subject" id="topicSubject" required>

<option value="">
Select Subject
</option>

{% for subject_name in all_subjects %}

<option value="{{ subject_name }}">
{{ subject_name }}
</option>

{% endfor %}

</select>

</div>


<div>

<label>Topic</label>

<select name="topic" id="topic" required>

<option value="">
Select Topic
</option>

</select>

</div>


<div>

<label>Topic Score</label>

<input
type="number"
name="score"
min="0"
max="100"
step="0.1"
placeholder="Example: 75"
required>

</div>


<button type="submit">
💾 Save Topic Progress
</button>

</form>

</div>


<div class="card">

<h3>📌 Topic Status Guide</h3>

<p>
<span class="topic-status status-strong">
80%+ Strong
</span>
</p>

<p>
<span class="topic-status status-moderate">
60-79% Moderate
</span>
</p>

<p>
<span class="topic-status status-weak">
Below 60% Weak
</span>
</p>

<p class="small">
Weak topics par extra PYQ practice aur revision karo.
</p>

</div>

</div>


<div class="card" style="margin-top:18px;">

<h3>📋 Topic History</h3>

<div class="table-wrapper">

<table>

<thead>

<tr>

<th>Date</th>
<th>Student</th>
<th>Exam</th>
<th>Subject</th>
<th>Topic</th>
<th>Score</th>
<th>Status</th>

</tr>

</thead>

<tbody>

{% for row in topics %}

<tr>

<td>
{{ row.date_time }}
</td>

<td>
{{ row.student_name }}
</td>

<td>
{{ row.exam }}
</td>

<td>
{{ row.subject }}
</td>

<td>
{{ row.topic }}
</td>

<td>
{{ row.score }}%
</td>

<td>

{% if row.score >= 80 %}

<span class="topic-status status-strong">
Strong
</span>

{% elif row.score >= 60 %}

<span class="topic-status status-moderate">
Moderate
</span>

{% else %}

<span class="topic-status status-weak">
Weak
</span>

{% endif %}

</td>

</tr>

{% else %}

<tr>

<td colspan="7">
No topic records yet.
</td>

</tr>

{% endfor %}

</tbody>

</table>

</div>

</div>

</section>


<!-- =====================================================
     HISTORY
====================================================== -->

<section class="section" id="history">

<div class="section-title">

<h2>📈 Performance History</h2>

<p>
Previous AI analysis records.
</p>

</div>


<div class="card">

<div class="chart-container">

<canvas id="historyChart"></canvas>

</div>

</div>


<div class="card" style="margin-top:18px;">

<div style="
display:flex;
justify-content:space-between;
align-items:center;
gap:10px;
flex-wrap:wrap;
">

<h3>
🗂️ Test History
</h3>

<a
class="btn btn-danger"
href="/clear_history"
onclick="
return confirm('Are you sure you want to clear all test history?')
"
>
Clear Test History
</a>

</div>


<div class="table-wrapper">

<table>

<thead>

<tr>

<th>Date</th>
<th>Student</th>
<th>Exam</th>
<th>Subject</th>
<th>Accuracy</th>
<th>Prediction</th>
<th>Level</th>

</tr>

</thead>

<tbody>

{% for row in history %}

<tr>

<td>
{{ row.date_time }}
</td>

<td>
{{ row.student_name }}
</td>

<td>
{{ row.exam }}
</td>

<td>
{{ row.subject }}
</td>

<td>
{{ row.accuracy }}%
</td>

<td>
{{ row.prediction }}%
</td>

<td>
{{ row.level }}
</td>

</tr>

{% else %}

<tr>

<td colspan="7">
No performance records yet.
</td>

</tr>

{% endfor %}

</tbody>

</table>

</div>

</div>

</section>


<footer class="footer">

AI Government Exam Performance Analyzer

<br>

<span class="small">
AI-Based Learning Pattern Analyzer | B.Tech Project 2026-27
</span>

</footer>


</main>


<script>

const examSubjects = {{ exam_subjects | tojson }};

const subjectTopics = {{ subject_topics | tojson }};


// =====================================================
// EXAM -> SUBJECT
// =====================================================

const examSelect = document.getElementById("exam");
const subjectSelect = document.getElementById("subject");

function updateSubjects() {

    if (!examSelect || !subjectSelect) {
        return;
    }

    const exam = examSelect.value;

    subjectSelect.innerHTML =
        '<option value="">Select Subject</option>';

    if (examSubjects[exam]) {

        examSubjects[exam].forEach(function(subject) {

            const option =
                document.createElement("option");

            option.value = subject;
            option.textContent = subject;

            subjectSelect.appendChild(option);

        });

    }
}


if (examSelect) {

    examSelect.addEventListener(
        "change",
        updateSubjects
    );

}


// =====================================================
// TOPIC -> SUBJECT
// =====================================================

const topicSubject =
    document.getElementById("topicSubject");

const topicSelect =
    document.getElementById("topic");


function updateTopics() {

    if (!topicSubject || !topicSelect) {
        return;
    }

    const subject = topicSubject.value;

    topicSelect.innerHTML =
        '<option value="">Select Topic</option>';

    if (subjectTopics[subject]) {

        subjectTopics[subject].forEach(function(topic) {

            const option =
                document.createElement("option");

            option.value = topic;
            option.textContent = topic;

            topicSelect.appendChild(option);

        });

    }

}


if (topicSubject) {

    topicSubject.addEventListener(
        "change",
        updateTopics
    );

}


// =====================================================
// AUTO CALCULATE WRONG + ACCURACY
// =====================================================

const attemptedInput =
    document.getElementById("questions_attempted");

const correctInput =
    document.getElementById("correct_answers");

const wrongInput =
    document.getElementById("wrong_answers");

const accuracyInput =
    document.getElementById("accuracy");


function calculateAccuracy() {

    if (!attemptedInput ||
        !correctInput ||
        !wrongInput ||
        !accuracyInput) {

        return;
    }

    let attempted =
        parseInt(attemptedInput.value) || 0;

    let correct =
        parseInt(correctInput.value) || 0;

    if (correct > attempted) {
        correct = attempted;
        correctInput.value = correct;
    }

    let wrong =
        attempted - correct;

    let accuracy = 0;

    if (attempted > 0) {
        accuracy =
            (correct / attempted) * 100;
    }

    wrongInput.value = wrong;
    accuracyInput.value =
        accuracy.toFixed(2);
}


if (attemptedInput) {
    attemptedInput.addEventListener(
        "input",
        calculateAccuracy
    );
}

if (correctInput) {
    correctInput.addEventListener(
        "input",
        calculateAccuracy
    );
}


// =====================================================
// SETTINGS
// =====================================================

const theme =
    document.getElementById("theme");

const fontSize =
    document.getElementById("fontSize");

const layout =
    document.getElementById("layout");


function applySettings() {

    const savedTheme =
        localStorage.getItem("theme") || "light";

    const savedFont =
        localStorage.getItem("fontSize") || "normal";

    const savedLayout =
        localStorage.getItem("layout") || "comfortable";


    document.body.classList.remove(
        "dark",
        "blue",
        "green"
    );

    if (savedTheme !== "light") {
        document.body.classList.add(savedTheme);
    }


    document.body.classList.remove(
        "small-font",
        "large-font"
    );

    if (savedFont === "small") {
        document.body.classList.add("small-font");
    }

    if (savedFont === "large") {
        document.body.classList.add("large-font");
    }


    document.body.classList.remove("compact");

    if (savedLayout === "compact") {
        document.body.classList.add("compact");
    }


    if (theme) {
        theme.value = savedTheme;
    }

    if (fontSize) {
        fontSize.value = savedFont;
    }

    if (layout) {
        layout.value = savedLayout;
    }
}


if (theme) {

    theme.addEventListener(
        "change",
        function() {

            localStorage.setItem(
                "theme",
                theme.value
            );

            applySettings();

        }
    );

}


if (fontSize) {

    fontSize.addEventListener(
        "change",
        function() {

            localStorage.setItem(
                "fontSize",
                fontSize.value
            );

            applySettings();

        }
    );

}


if (layout) {

    layout.addEventListener(
        "change",
        function() {

            localStorage.setItem(
                "layout",
                layout.value
            );

            applySettings();

        }
    );

}


applySettings();


// =====================================================
// CURRENT PERFORMANCE CHART
// =====================================================

{% if result %}

const currentCanvas =
    document.getElementById("currentChart");

if (currentCanvas) {

    new Chart(currentCanvas, {

        type: "bar",

        data: {

            labels: [
                "AI Prediction",
                "Mock Score",
                "Accuracy"
            ],

            datasets: [{

                label: "Performance %",

                data: [
                    {{ result.prediction }},
                    {{ result.mock_score }},
                    {{ result.accuracy }}
                ]

            }]

        },

        options: {

            responsive: true,

            maintainAspectRatio: false,

            scales: {

                y: {
                    beginAtZero: true,
                    max: 100
                }

            }

        }

    });

}

{% endif %}


// =====================================================
// HISTORY CHART
// =====================================================

const historyCanvas =
    document.getElementById("historyChart");


if (historyCanvas) {

    const historyLabels =
        {{ history_labels | tojson }};

    const historyPredictions =
        {{ history_predictions | tojson }};


    new Chart(historyCanvas, {

        type: "line",

        data: {

            labels: historyLabels,

            datasets: [{

                label: "AI Prediction %",

                data: historyPredictions,

                tension: 0.3,

                fill: false

            }]

        },

        options: {

            responsive: true,

            maintainAspectRatio: false,

            scales: {

                y: {

                    beginAtZero: true,

                    max: 100

                }

            }

        }

    });

}

</script>


</body>

</html>
"""


# =========================================================
# HOME ROUTE
# =========================================================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    form_data = None

    if request.method == "POST":

        try:

            student_name = request.form.get(
                "student_name",
                ""
            ).strip()

            exam = request.form.get(
                "exam",
                ""
            ).strip()

            subject = request.form.get(
                "subject",
                ""
            ).strip()

            study_hours = float(
                request.form.get(
                    "study_hours",
                    0
                )
            )

            mock_score = float(
                request.form.get(
                    "mock_score",
                    0
                )
            )

            questions_attempted = int(
                request.form.get(
                    "questions_attempted",
                    0
                )
            )

            correct_answers = int(
                request.form.get(
                    "correct_answers",
                    0
                )
            )

            # Safety checks

            study_hours = max(
                0,
                study_hours
            )

            mock_score = max(
                0,
                min(100, mock_score)
            )

            questions_attempted = max(
                0,
                questions_attempted
            )

            correct_answers = max(
                0,
                correct_answers
            )

            if correct_answers > questions_attempted:
                correct_answers = questions_attempted

            wrong_answers = (
                questions_attempted -
                correct_answers
            )

            if questions_attempted > 0:

                accuracy = (
                    correct_answers /
                    questions_attempted
                ) * 100

            else:

                accuracy = 0


            accuracy = round(
                accuracy,
                2
            )


            prediction = predict_performance(
                study_hours,
                mock_score,
                questions_attempted,
                correct_answers,
                wrong_answers,
                accuracy
            )


            level = get_level(
                prediction
            )


            analysis = subject_analysis(
                accuracy
            )


            recommendations = generate_recommendations(
                study_hours,
                mock_score,
                questions_attempted,
                correct_answers,
                wrong_answers,
                accuracy
            )


            weak_area = detect_weak_area(
                study_hours,
                mock_score,
                accuracy,
                subject
            )


            study_plan = create_study_plan(
                study_hours,
                accuracy,
                mock_score
            )


            badges = get_badges(
                accuracy,
                prediction,
                questions_attempted,
                mock_score,
                study_hours
            )


            date_time = datetime.now().strftime(
                "%d-%m-%Y %I:%M %p"
            )


            conn = get_db()

            cur = conn.cursor()

            cur.execute("""
                INSERT INTO results (
                    student_name,
                    exam,
                    subject,
                    study_hours,
                    mock_score,
                    questions_attempted,
                    correct_answers,
                    wrong_answers,
                    accuracy,
                    prediction,
                    level,
                    subject_analysis,
                    date_time
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                student_name,
                exam,
                subject,
                study_hours,
                mock_score,
                questions_attempted,
                correct_answers,
                wrong_answers,
                accuracy,
                prediction,
                level,
                analysis,
                date_time
            ))

            report_id = cur.lastrowid

            conn.commit()

            conn.close()


            result = {
                "id": report_id,
                "student_name": student_name,
                "exam": exam,
                "subject": subject,
                "study_hours": study_hours,
                "mock_score": mock_score,
                "questions_attempted": questions_attempted,
                "correct_answers": correct_answers,
                "wrong_answers": wrong_answers,
                "accuracy": accuracy,
                "prediction": prediction,
                "level": level,
                "subject_analysis": analysis,
                "recommendations": recommendations,
                "weak_area": weak_area,
                "study_plan": study_plan,
                "badges": badges
            }


            form_data = {
                "student_name": student_name,
                "exam": exam,
                "subject": subject,
                "study_hours": study_hours,
                "mock_score": mock_score,
                "questions_attempted": questions_attempted,
                "correct_answers": correct_answers,
                "wrong_answers": wrong_answers,
                "accuracy": accuracy
            }


        except Exception as e:

            return f"""
            <h2>❌ Error</h2>
            <p>{str(e)}</p>
            <p>
            Go back and check your input values.
            </p>
            """


    dashboard = get_dashboard_data()

    subjects = get_subject_progress()

    history = get_history()

    topics = get_topic_history()


    # Oldest -> newest for graph

    history_for_chart = list(reversed(history))

    history_labels = [
        row["date_time"]
        for row in history_for_chart
    ]

    history_predictions = [
        round(row["prediction"], 2)
        for row in history_for_chart
    ]


    all_subjects = sorted(
        set(
            subject
            for subjects_list in EXAM_SUBJECTS.values()
            for subject in subjects_list
        )
    )


    return render_template_string(
        HTML,

        result=result,

        form_data=form_data,

        exams=list(
            EXAM_SUBJECTS.keys()
        ),

        exam_subjects=EXAM_SUBJECTS,

        subject_topics=SUBJECT_TOPICS,

        all_subjects=all_subjects,

        dashboard=dashboard,

        subjects=subjects,

        history=history,

        topics=topics,

        history_labels=history_labels,

        history_predictions=history_predictions
    )


# =========================================================
# SAVE TOPIC
# =========================================================

@app.route(
    "/save_topic",
    methods=["POST"]
)
def save_topic():

    student_name = request.form.get(
        "student_name",
        ""
    ).strip()

    exam = request.form.get(
        "exam",
        ""
    ).strip()

    subject = request.form.get(
        "subject",
        ""
    ).strip()

    topic = request.form.get(
        "topic",
        ""
    ).strip()

    try:
        score = float(
            request.form.get(
                "score",
                0
            )
        )
    except Exception:
        score = 0


    score = max(
        0,
        min(100, score)
    )


    date_time = datetime.now().strftime(
        "%d-%m-%Y %I:%M %p"
    )


    conn = get_db()

    conn.execute("""
        INSERT INTO topics (
            student_name,
            exam,
            subject,
            topic,
            score,
            date_time
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        student_name,
        exam,
        subject,
        topic,
        score,
        date_time
    ))

    conn.commit()

    conn.close()


    return redirect(
        url_for("home") + "#topics"
    )


# =========================================================
# DOWNLOAD REPORT
# =========================================================

@app.route(
    "/download_report/<int:report_id>"
)
def download_report(report_id):

    conn = get_db()

    row = conn.execute("""
        SELECT *
        FROM results
        WHERE id = ?
    """, (report_id,)).fetchone()

    conn.close()


    if not row:

        return "Report not found."


    recommendations = generate_recommendations(
        row["study_hours"],
        row["mock_score"],
        row["questions_attempted"],
        row["correct_answers"],
        row["wrong_answers"],
        row["accuracy"]
    )


    study_plan = create_study_plan(
        row["study_hours"],
        row["accuracy"],
        row["mock_score"]
    )


    report = []

    report.append(
        "AI GOVERNMENT EXAM PERFORMANCE ANALYZER"
    )

    report.append(
        "=" * 55
    )

    report.append("")

    report.append(
        f"Student Name : {row['student_name']}"
    )

    report.append(
        f"Exam         : {row['exam']}"
    )

    report.append(
        f"Subject      : {row['subject']}"
    )

    report.append(
        f"Date         : {row['date_time']}"
    )

    report.append("")

    report.append(
        "PERFORMANCE"
    )

    report.append(
        "-" * 55
    )

    report.append(
        f"Study Hours          : {row['study_hours']}"
    )

    report.append(
        f"Mock Test Score      : {row['mock_score']}%"
    )

    report.append(
        f"Questions Attempted  : {row['questions_attempted']}"
    )

    report.append(
        f"Correct Answers      : {row['correct_answers']}"
    )

    report.append(
        f"Wrong Answers        : {row['wrong_answers']}"
    )

    report.append(
        f"Accuracy             : {row['accuracy']}%"
    )

    report.append(
        f"AI Prediction        : {row['prediction']}%"
    )

    report.append(
        f"Performance Level    : {row['level']}"
    )

    report.append(
        f"Subject Analysis     : {row['subject_analysis']}"
    )

    report.append("")

    report.append(
        "SMART RECOMMENDATIONS"
    )

    report.append(
        "-" * 55
    )

    for i, recommendation in enumerate(
        recommendations,
        start=1
    ):

        report.append(
            f"{i}. {recommendation}"
        )


    report.append("")

    report.append(
        "PERSONALIZED STUDY PLAN"
    )

    report.append(
        "-" * 55
    )

    for item in study_plan:

        report.append(
            f"- {item['task']} | {item['time']}"
        )

        report.append(
            f"  {item['description']}"
        )


    report.append("")

    report.append(
        "Generated by AI-Based Learning Pattern Analyzer"
    )


    report_text = "\n".join(
        report
    )


    file_stream = io.BytesIO()

    file_stream.write(
        report_text.encode("utf-8")
    )

    file_stream.seek(0)


    filename = (
        "AI_Performance_Report_"
        + str(row["id"])
        + ".txt"
    )


    return send_file(
        file_stream,
        as_attachment=True,
        download_name=filename,
        mimetype="text/plain"
    )


# =========================================================
# CLEAR HISTORY
# =========================================================

@app.route("/clear_history")
def clear_history():

    conn = get_db()

    conn.execute(
        "DELETE FROM results"
    )

    conn.commit()

    conn.close()


    return redirect(
        url_for("home") + "#history"
    )


# =========================================================
# RUN APP
# =========================================================

if __name__ == "__main__":

    print("")
    print("=" * 60)
    print("AI GOVERNMENT EXAM PERFORMANCE ANALYZER")
    print("=" * 60)
    print("")

    if model is not None:
        print("✅ AI Model loaded successfully")
    else:
        print(
            "⚠️ AI Model not found. "
            "Fallback prediction will be used."
        )

    print("")
    print(
        "🌐 Open: http://127.0.0.1:5000"
    )
    print("")

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )