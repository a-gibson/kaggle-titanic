#!/usr/bin/env python3

import os

# import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier

data_path = "./data"
prediction_path = "./predictions"
prediction_csv = os.path.join(prediction_path, 'my_submission.csv')

for dirname, _, filenames in os.walk(data_path):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Read in training data
train_data = pd.read_csv(os.path.join(data_path, "train.csv"))

# Read in test data
test_data = pd.read_csv(os.path.join(data_path, "test.csv"))

# Calculate what percentage of women survived
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)

# Calculate what percentage of men survived
men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)

# Apply the Random Forest method to the training data, as per the example
y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame(
    {'PassengerId': test_data.PassengerId, 'Survived': predictions}
)

output.to_csv(prediction_csv, index=False)

print("Prediction saved to:\n{}".format(prediction_csv))
