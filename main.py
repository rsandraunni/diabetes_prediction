import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Load dataset
df = pd.read_csv("data/diabetes.csv")

# Data cleaning
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols] = df[cols].replace(0, df[cols].median())

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(class_weight='balanced', probability=True))
])

# Tuning
param_grid = {
    "svm__kernel": ["linear", "rbf", "poly"],
    "svm__C": [0.1, 1, 10],
    "svm__gamma": ["scale", "auto"]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="recall")
grid.fit(X_train, y_train)

model = grid.best_estimator_

# -------- USER INPUT --------
features = list(X.columns)

print("\nEnter values for prediction:")
values = []

for f in features:
    val = float(input(f"{f}: "))
    values.append(val)

input_df = pd.DataFrame([values], columns=features)

prediction = model.predict(input_df)[0]
prob = model.predict_proba(input_df)[0][1]

result = "Diabetic" if prediction == 1 else "Not Diabetic"

print("\nPrediction:", result)
print("Confidence:", round(prob * 100, 2), "%")