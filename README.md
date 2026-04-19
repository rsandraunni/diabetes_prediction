🩺** Diabetes Prediction using SVM**

📌Project Overview

This project focuses on predicting whether a person is diabetic or not using a Support Vector Machine (SVM) model. The model is trained on medical diagnostic data and makes predictions based on user input features.

📊 Dataset

The dataset used is the Pima Indians Diabetes Dataset, which contains medical details of patients.

Features used:
Pregnancies
Glucose
Blood Pressure
Skin Thickness
Insulin
BMI (Body Mass Index)
Diabetes Pedigree Function
Age
Target:
0 → Not Diabetic
1 → Diabetic
⚙️ Technologies Used
Python
Pandas
NumPy
Scikit-learn
🚀 How It Works
Load the dataset (diabetes.csv)
Perform basic data analysis
Split data into training and testing sets
Train an SVM model
Evaluate using:
Accuracy
Precision
Recall
Take user input and predict diabetes status
🧠 Machine Learning Model
Algorithm: Support Vector Machine (SVM)
Kernel used: Polynomial (can be changed to RBF for better performance)
📈 Model Evaluation

The model is evaluated using:

Training Accuracy
Testing Accuracy
Precision
Recall


🖥️ How to Run

Step 1: Clone the repository
git clone <your-repo-link>
cd <your-repo-folder>

Step 2: Install dependencies
pip install pandas numpy scikit-learn

Step 3: Add dataset
Make sure diabetes.csv is present in the project directory
OR update the file path in the code.

Step 4: Run the script
python your_script_name.py

🧪 Example Input

Pregnancies: 2  
Glucose: 120  
Blood Pressure: 70  
Skin Thickness: 20  
Insulin: 85  
BMI: 28.5  
Diabetes Pedigree Function: 0.5  
Age: 30  
Output:
The person is: Not Diabetic

⚠️ Limitations

Model accuracy depends on dataset quality
No feature scaling applied (can be improved)
No advanced preprocessing

🔧 Future Improvements

Add feature scaling (StandardScaler)
Try different kernels (RBF, Linear)
Add GUI or web interface
Improve accuracy with hyperparameter tuning

👤 Author

R. Sandra Unni

📌 Note

This project is for educational purposes and should not be used for real medical diagnosis.
