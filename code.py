import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")
gender_submission = pd.read_csv("gender_submission.csv")

women = train.loc[train.Sex == "female"]["Survived"]
rate_women = sum(women) / len(women)
# print(rate_women)

men = train.loc[train.Sex == "male"]["Survived"]
rate_men = sum(men) / len(men)
# print(rate_men)

y = train["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train[features])
X_test = pd.get_dummies(test[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame(
    {"PassengerId": test.PassengerId, "Survived": predictions}
)

print(output)
output.to_csv("submission.csv", index=False)

print("Your submission was successfuly saved!")
