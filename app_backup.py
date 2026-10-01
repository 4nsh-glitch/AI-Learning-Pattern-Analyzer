from flask import Flask, request, render_template_string
import joblib

app = Flask(__name__)

model = joblib.load("learning_pattern_model.pkl")

HTML = """
<!DOCTYPE html>
<html>
<head>

<title>AI Government Exam Performance Analyzer</title>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>

body {
    font-family: Arial, sans-serif;
    background: #f1f5f9;
    margin: 0;
    padding: 40px;
}

.container {
    max-width: 600px;
    margin: auto;
    background: white;
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

h1 {
    text-align: center;
}

label {
    font-weight: bold;
}

input, select {
    width: 100%;
    padding: 11px;
    margin-top: 6px;
    margin-bottom: 15px;
    border: 1px solid #ccc;
    border-radius: 6px;
    box-sizing: border-box;
}

button {
    width: 100%;
    padding: 13px;
    border: none;
    border-radius: 7px;
    background: #2563eb;
    color: white;
    font-size: 16px;
    cursor: pointer;
}

button:hover {
    background: #1d4ed8;
}

.result {
    margin-top: 25px;
    padding: 20px;
    background: #f8fafc;
    border-radius: 10px;
}

.score-card {
    text-align: center;
    padding: 25px;
    border-radius: 12px;
    background: #eff6ff;
}

.score {
    font-size: 48px;
    font-weight: bold;
}

.level {
    display: inline-block;
    padding: 8px 18px;
    border-radius: 20px;
    background: #2563eb;
    color: white;
    font-weight: bold;
}

.recommendation {
    margin-top: 20px;
    line-height: 1.7;
}

</style>

<script>

function updateSubjects() {

    let exam = document.getElementById("exam").value;
    let subject = document.getElementById("subject");

    subject.innerHTML = "";

    let subjects = [];

    if (exam === "SSC CGL") {
        subjects = ["Maths", "Reasoning", "English", "GK"];
    }

    else if (exam === "SSC CHSL") {
        subjects = ["Maths", "Reasoning", "English", "GK"];
    }

    else if (exam === "SSC GD") {
        subjects = ["GK", "General Science", "Reasoning", "Maths"];
    }

    else if (exam === "SSC CPO") {
        subjects = ["Maths", "Reasoning", "English", "GK"];
    }

    else if (exam === "Railway NTPC") {
        subjects = ["Maths", "Reasoning", "General Science", "GK"];
    }

    else if (exam === "Railway Group D") {
        subjects = ["Maths", "Reasoning", "General Science", "GK"];
    }

    subjects.forEach(function(item) {

        let option = document.createElement("option");

        option.value = item;
        option.text = item;

        subject.appendChild(option);

    });

}

function calculateAccuracy() {

    let attempted =
        Number(document.getElementById("questions_attempted").value);

    let correct =
        Number(document.getElementById("correct_answers").value);

    let accuracy = 0;

    if (attempted > 0) {
        accuracy = (correct / attempted) * 100;
    }

    let wrong = attempted - correct;

    document.getElementById("wrong_answers").value =
        wrong >= 0 ? wrong : 0;

    document.getElementById("accuracy").value =
        accuracy.toFixed(2);
} 

</script>

</head>

<body>

<div class="container">

<h1>🎯 AI Government Exam Performance Analyzer</h1>

<form method="POST">

<label>Student Name</label>

<input type="text" 
name="student_name"
placeholder="Enter your name"
required>


<label>Select Exam</label>

<select name="exam"
id="exam"
onchange="updateSubjects()"
required>

<option value="">-- Select Exam --</option>

<option value="SSC CGL">SSC CGL</option>

<option value="SSC CHSL">SSC CHSL</option>

<option value="SSC GD">SSC GD</option>

<option value="SSC CPO">SSC CPO</option>

<option value="Railway NTPC">Railway NTPC</option>

<option value="Railway Group D">Railway Group D</option>

</select>


<label>Select Subject</label>

<select name="subject"
id="subject"
required>

<option value="">-- Select Exam First --</option>

</select>


<label>Study Hours per Day</label>

<input type="number"
name="study_hours"
step="0.1"
min="0"
max="24"
required>


<label>Mock Test Score</label>

<input type="number"
name="mock_score"
step="0.1"
min="0"
max="100"
required>


<label>Questions Attempted</label>

<input type="number"
name="questions_attempted"
id="questions_attempted"
min="1"
oninput="calculateAccuracy()"
required>


<label>Correct Answers</label>

<input type="number"
name="correct_answers"
id="correct_answers"
min="0"
oninput="calculateAccuracy()"
required>


<label>Wrong Answers (Automatic)</label>

<input type="number"
name="wrong_answers"
id="wrong_answers"
readonly>


<label>Accuracy (%)</label>

<input type="number"
name="accuracy"
id="accuracy"
readonly>


<button type="submit">
Analyze Exam Performance
</button>

</form>


{% if prediction is not none %}

<div class="result">

<h2>📊 Government Exam Performance Report</h2>

<p>
<strong>Student:</strong> {{ student_name }}
</p>

<p>
<strong>Exam:</strong> {{ exam }}
</p>

<p>
<strong>Subject:</strong> {{ subject }}
</p>


<div class="score-card">

<h3>Predicted Performance</h3>

<div class="score">
{{ prediction }}
</div>

<p>out of 100</p>

<div class="level">
{{ level }}
</div>

</div>

<div style="margin-top: 25px;">

    <h3>📈 Performance Analysis</h3>

    <canvas id="performanceChart"></canvas>

</div>

<script>

const ctx = document.getElementById("performanceChart");

new Chart(ctx, {
    type: "bar",

    data: {
        labels: [
            "Predicted Performance",
            "Mock Score",
            "Accuracy"
        ],

        datasets: [{
            label: "Score (%)",

            data: [
                {{ prediction }},
                {{ mock_score }},
                {{ accuracy }}
            ]
        }]
    },

    options: {
        responsive: true,

        scales: {
            y: {
                beginAtZero: true,
                max: 100
            }
        }
    }
});

</script>

<div class="recommendation">

<h3>📚 Recommendations</h3>

<ul>

{% for recommendation in recommendations %}

<li>{{ recommendation }}</li>

{% endfor %}

</ul>

</div>

</div>

{% endif %}

</div>

</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    student_name = ""
    exam = ""
    subject = ""

    level = ""

    recommendations = []

    mock_score = 0
    accuracy = 0

    if request.method == "POST":

        student_name = request.form["student_name"]

        exam = request.form["exam"]

        subject = request.form["subject"]

        study_hours = float(
            request.form["study_hours"]
        )

        mock_score = float(
            request.form["mock_score"]
        )

        questions_attempted = float(
            request.form["questions_attempted"]
        )

        correct_answers = float(
            request.form["correct_answers"]
        )

        wrong_answers = float(
            request.form["wrong_answers"]
        )

        accuracy = float(
            request.form["accuracy"]
        )


        features = [[

            study_hours,
            mock_score,
            questions_attempted,
            correct_answers,
            wrong_answers,
            accuracy

        ]]


        prediction = round(
            model.predict(features)[0],
            2
        )


        if prediction >= 80:

            level = "Excellent"

        elif prediction >= 60:

            level = "Good"

        elif prediction >= 40:

            level = "Needs Improvement"

        else:

            level = "Low"


        if study_hours < 3:

            recommendations.append(
                "Increase daily study time to at least 3-4 hours."
            )


        if accuracy < 60:

            recommendations.append(
                "Improve accuracy by practicing more questions."
            )


        if wrong_answers > 15:

            recommendations.append(
                "Reduce wrong attempts and focus on question selection."
            )


        if mock_score < 60:

            recommendations.append(
                "Take regular mock tests and revise weak topics."
            )


        if correct_answers < questions_attempted * 0.7:

            recommendations.append(
                "Practice more PYQs for this subject."
            )

            

            if not recommendations:

              recommendations.append(
                "Your preparation is going well. Continue regular mock tests and revision."
            )


    return render_template_string(

        HTML,

        prediction=prediction,

        student_name=student_name,

        exam=exam,

        subject=subject,

        level=level,

        recommendations=recommendations,

        mock_score=mock_score,

        accuracy=accuracy

    ) 
if __name__ == "__main__":
    app.run(debug=True)