import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/Titanic.csv')

data.head()

data.info()

print(data.head())
print(data.columns)

data = data.fillna(method='ffill')

# Handle missing values
data = data.fillna(method='ffill')

# Convert categorical variables into numerical
categorical_columns = ['Sex', 'Embarked', 'Pclass']
for col in categorical_columns:
    if col in data.columns:
        data = pd.get_dummies(data, columns=[col], drop_first=True)

  # Drop irrelevant columns if they exist
columns_to_drop = ['Name', 'Ticket', 'Cabin']
for col in columns_to_drop:
    if col in data.columns:
        data.drop(col, axis=1, inplace=True)

  # Ensure all data is numeric
data = data.apply(pd.to_numeric, errors='coerce')

# Visualize the distribution of survivors
sns.countplot(x='survived', data=data)
plt.title('Survival Distribution')
plt.show()

# Correlation matrix to see relationships between features
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, fmt='.2f')
plt.title('Correlation Matrix')
plt.show()



# Model Building
X = data.drop('survived', axis=1)
y = data['survived']

# Impute missing values using mean strategy
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Train Random Forest model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Evaluate models
accuracy_lr = accuracy_score(y_test, y_pred_lr)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(f"Logistic Regression Accuracy: {accuracy_lr}")
print(f"Random Forest Accuracy: {accuracy_rf}")

print("Logistic Regression Classification Report")
print(classification_report(y_test, y_pred_lr))

print("Random Forest Classification Report")
print(classification_report(y_test, y_pred_rf))

# Predictions are already made during evaluation (y_pred_lr, y_pred_rf)

if accuracy_rf > accuracy_lr:
    print("Random Forest performs better.")
else:
    print("Logistic Regression performs better.")




