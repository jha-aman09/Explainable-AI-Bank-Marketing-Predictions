!pip install lime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

data = pd.read_csv("/content/drive/MyDrive/bank-additional-full.csv", sep=";")

data['y'] = data['y'].map({'yes': 1, 'no': 0})

data_encoded = pd.get_dummies(data, drop_first=True)

X = data_encoded.drop('y', axis=1)
y = data_encoded['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

importances = model.feature_importances_
feature_names = X.columns
sorted_indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Global Feature Importance")
plt.bar(range(X.shape[1]), importances[sorted_indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[sorted_indices], rotation=90)
plt.tight_layout()
plt.show()

feature_names = X.columns
explainer = LimeTabularExplainer(
    X_train.values,
    training_labels=y_train.values,
    feature_names=feature_names,
    class_names=["no", "yes"],
    mode="classification"
)

obs_4 = X_test.iloc[3].values
obs_20 = X_test.iloc[19].values

exp_4 = explainer.explain_instance(obs_4, model.predict_proba, num_features=10)
exp_20 = explainer.explain_instance(obs_20, model.predict_proba, num_features=10)

print("Explanation for Observation #4")
exp_4.show_in_notebook()
exp_4.as_pyplot_figure()
plt.show()

print("Explanation for Observation #20")
exp_20.show_in_notebook()
exp_20.as_pyplot_figure()
plt.show()