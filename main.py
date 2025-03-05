import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
import numpy as np

# Load dataset
file_path = "diabetes.csv"  # Update the path if needed
ds = pd.read_csv(file_path)

# Display first 10 rows
print("First 10 rows of dataset:")
print(ds.head(10))

# Print dataset shape
print("\nDataset shape:", ds.shape)

# Describe dataset
print("\nDataset summary:")
print(ds.describe())

# Define features and target
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
x = ds[features]  # Features
y = ds['Outcome']  # Target variable

# Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=40)

# Create SVM classifier with polynomial kernel
clf = svm.SVC(kernel='poly')

# Train the classifier
clf.fit(x_train, y_train)

# Make predictions
y_pred_train = clf.predict(x_train)
y_pred_test = clf.predict(x_test)

# Calculate and print model performance metrics
print("\nTraining Accuracy:", metrics.accuracy_score(y_train, y_pred_train))
print("Testing Accuracy:", metrics.accuracy_score(y_test, y_pred_test))
print("Precision:", metrics.precision_score(y_test, y_pred_test))
print("Recall:", metrics.recall_score(y_test, y_pred_test))

# Take user input for new sample
print("\nEnter values for prediction:")
pregnancies = float(input("Pregnancies: "))
glucose = float(input("Glucose: "))
blood_pressure = float(input("Blood Pressure: "))
skin_thickness = float(input("Skin Thickness: "))
insulin = float(input("Insulin: "))
bmi = float(input("BMI: "))
dpf = float(input("Diabetes Pedigree Function: "))
age = float(input("Age: "))

# Create input DataFrame with feature names
X1 = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
X1_df = pd.DataFrame(X1, columns=features)  # Convert to DataFrame

# Predict the outcome
new_prediction = clf.predict(X1_df)
result = "Diabetic" if new_prediction[0] == 1 else "Not Diabetic"
print("\nThe person is:", result)
