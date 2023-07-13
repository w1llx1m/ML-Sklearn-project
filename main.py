import pandas as pd
from sklearn.tree import DecisionTreeClassifier


sleep_data = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
X = sleep_data.drop(columns=['Sleep Disorder'])
y = sleep_data['Sleep Disorder']


model = DecisionTreeClassifier()
model.fit(X, y)

#predictions = model.predict([ [1, "Male", 21, "Other", 8.0, 8, 90, 8, "Normal", 130/85] ])
#temp

print(X)