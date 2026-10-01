import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load government exam dataset
data = pd.read_csv("data/student_data.csv")

# Features
X = data[
    [
        "study_hours",
        "mock_score",
        "questions_attempted",
        "correct_answers",
        "wrong_answers",
        "accuracy"
    ]
]

# Target
y = data["performance_score"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create ML model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Error check
mae = mean_absolute_error(y_test, predictions)

print("Government Exam ML Model trained successfully!")
print("Mean Absolute Error:", round(mae, 2))

# Save model
joblib.dump(model, "learning_pattern_model.pkl")

print("Model saved successfully!")