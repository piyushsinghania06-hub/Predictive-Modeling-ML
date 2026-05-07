import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc

# Load Dataset
dataset = pd.read_csv('customer_data.csv')

# Display Dataset
print(dataset.head())

# Remove Missing Values
dataset = dataset.dropna()

# Encode Categorical Values
encoder = LabelEncoder()
dataset['Gender'] = encoder.fit_transform(dataset['Gender'])

# Features and Target
X = dataset[['Age', 'Salary', 'Gender']]
y = dataset['Purchased']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Linear Regression
# -----------------------------
lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

print("\nLinear Regression Predictions:")
print(lr_predictions[:5])

# -----------------------------
# Decision Tree
# -----------------------------
dt_model = DecisionTreeClassifier()

dt_model.fit(X_train, y_train)

dt_predictions = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_predictions)

print("\nDecision Tree Accuracy:")
print(dt_accuracy)

# -----------------------------
# Random Forest
# -----------------------------
rf_model = RandomForestClassifier(n_estimators=100)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predictions)

print("\nRandom Forest Accuracy:")
print(rf_accuracy)

# -----------------------------
# Classification Report
# -----------------------------
print("\nClassification Report:")
print(classification_report(y_test, rf_predictions))

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, rf_predictions)

plt.figure(figsize=(6,4))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.show()

# -----------------------------
# ROC Curve
# -----------------------------
probabilities = rf_model.predict_proba(X_test)
probabilities = probabilities[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, probabilities)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, label='ROC Curve (area = %0.2f)' % roc_auc)

plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend()

plt.show()

# -----------------------------
# Accuracy Comparison Graph
# -----------------------------
models = ['Decision Tree', 'Random Forest']
accuracies = [dt_accuracy, rf_accuracy]

plt.figure(figsize=(6,4))

plt.bar(models, accuracies)

plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')

plt.show()
