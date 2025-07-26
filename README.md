🧑‍💻 My Role in the Project
I developed a suite of machine learning models focused on health prediction. Each model was built from the ground up, covering the entire lifecycle from raw data preprocessing to model evaluation and export for deployment.

1️⃣ Disease Prediction Model
Goal: Predict the most probable disease based on user-input symptoms.

Data: Multi-symptom records mapped to 40+ disease classes.

Preprocessing:

Combined multiple symptom columns

Cleaned and tokenized text inputs

Applied one-hot encoding using MultiLabelBinarizer

Models Implemented:

Random Forest (tuned with GridSearchCV)

K-Nearest Neighbors

Gradient Boosting

Evaluation:

10-fold cross-validation

Accuracy, macro F1-score, and confusion matrices

Enhancement: Merged disease descriptions and recommended precautions for informative output

2️⃣ Heart Disease Prediction Model
Goal: Classify the likelihood of heart disease using clinical features.

Data: Patient health records (e.g., age, cholesterol, blood pressure, etc.)

Preprocessing:

Handled missing values and scaled features

Performed correlation analysis and feature selection

Modeling:

Applied logistic regression, random forest, and gradient boosting

Compared model performance to find the best classifier

Output: Binary classification with interpretability metrics

3️⃣ Diabetes Prediction Model
Goal: Predict whether a patient is likely to develop diabetes.

Data: Medical measurements (e.g., glucose level, BMI, insulin)

Preprocessing:

Normalized input features

Analyzed class balance and applied stratified train/test split

Models Used:

Logistic Regression

KNN

Random Forest

Evaluation:

Confusion matrix, ROC-AUC curve, precision/recall

Achieved high accuracy and low false negatives

✅ Summary of Contributions
Built and optimized 3 distinct ML models for medical predictions

Designed complete data preprocessing pipelines tailored for each dataset

Conducted model comparison and performance tuning

Developed a prediction interface returning both output and medical advice

Exported trained models and encoders for integration into flluter app his name (DOCTORY)
