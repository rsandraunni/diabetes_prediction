import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
import numpy as np


file_path = "diabetes.csv" 
ds = pd.read_csv(file_path)


print("First 10 rows of dataset:")
print(ds.head(10))


print("\nDataset shape:", ds.shape)


print("\nDataset summary:")
print(ds.describe())


features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
x = ds[features]  # Features
y = ds['Outcome']  # Target variable


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=40)


clf = svm.SVC(kernel='poly')


clf.fit(x_train, y_train)


y_pred_train = clf.predict(x_train)
y_pred_test = clf.predict(x_test)


print("\nTraining Accuracy:", metrics.accuracy_score(y_train, y_pred_train))
print("Testing Accuracy:", metrics.accuracy_score(y_test, y_pred_test))
print("Precision:", metrics.precision_score(y_test, y_pred_test))
print("Recall:", metrics.recall_score(y_test, y_pred_test))


print("\nEnter values for prediction:")
pregnancies = float(input("Pregnancies: "))
glucose = float(input("Glucose: "))
blood_pressure = float(input("Blood Pressure: "))
skin_thickness = float(input("Skin Thickness: "))
insulin = float(input("Insulin: "))
bmi = float(input("BMI: "))
dpf = float(input("Diabetes Pedigree Function: "))
age = float(input("Age: "))


X1 = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
X1_df = pd.DataFrame(X1, columns=features)


new_prediction = clf.predict(X1_df)
result = "Diabetic" if new_prediction[0] == 1 else "Not Diabetic"
print("\nThe person is:", result)