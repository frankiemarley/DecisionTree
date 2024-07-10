
#     Pregnancies. Number of pregnancies of the patient (numeric)
#     Glucose. Plasma glucose concentration 2 hours after an oral glucose tolerance test (numeric)
#     BloodPressure. Diastolic blood pressure (measured in mm Hg) (numeric)
#     SkinThickness. Triceps skin fold thickness (measured in mm) (numeric)
#     Insulin. 2-hour serum insulin (measured in mu U/ml) (numeric)
#     BMI. Body mass index (numeric)
#     DiabetesPedigreeFunction. Diabetes Pedigree Function (numeric)
#     Age. Age of patient (numeric)
#     Outcome. Class variable (0 or 1), being 0 negative in diabetes and 1 positive (numeric)



import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import classification_report, mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import accuracy_score
from pickle import dump


# Load data
data = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv')

# print(data)
# print(data.info())
# print(data.describe)
# print(data.shape)


# Adjust display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Assuming 'df' is your DataFrame
summary = data.describe()

print(summary)


# Drop columns 'Pregnancies' and 'SkinThickness' from the dataset
data.drop(columns=['Pregnancies', 'SkinThickness'], inplace=True)

# Split data into X (features) and y (target variable) for classification
X_cls = data.drop(columns=['Outcome'])
y_cls = data['Outcome']

# Split into training and testing sets for classification
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

# Build Decision Tree Classifier
clf_model = DecisionTreeClassifier(random_state=42, max_depth=5)  # Adjust max_depth as needed for classification
clf_model.fit(X_train_cls, y_train_cls)

# Predict on test set for classification
y_pred_cls = clf_model.predict(X_test_cls)

# Evaluate accuracy for classification
accuracy_cls = accuracy_score(y_test_cls, y_pred_cls)
print("Classification Accuracy:", accuracy_cls)

# Plot feature importances for classification
plt.figure(figsize=(10, 6))
feature_importances = pd.Series(clf_model.feature_importances_, index=X_cls.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Important Features (Classification)')
plt.xlabel('Relative Importance')
plt.show()

# Plot the Decision Tree for classification (compact view)
plt.figure(figsize=(15, 8))
plot_tree(clf_model, feature_names=X_cls.columns, class_names=["0", "1"], filled=True, fontsize=10)
plt.title('Decision Tree Classifier')
plt.show()


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

hyperparams = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid = GridSearchCV(clf_model, hyperparams, scoring = "accuracy", cv = 10)
grid

grid.fit(X_train_cls, y_train_cls)

print(f"Best hyperparameters: {grid.best_params_}")

model = DecisionTreeClassifier(criterion = "entropy", max_depth = 5, min_samples_leaf = 4, min_samples_split = 2, random_state = 42)
model.fit(X_train_cls, y_train_cls)

y_pred = model.predict(X_test_cls)
y_pred

accuracy_cls2 = accuracy_score(y_test_cls, y_pred)
print("Classification Accuracy 2:", accuracy_cls2)