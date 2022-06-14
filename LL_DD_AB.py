import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('./Data/features_3_sec.csv')
data = data.iloc[0:, 1:]

Y = data['label']
X = data.loc[:, data.columns != 'label']

cols = X.columns
scaler = preprocessing.MinMaxScaler()
np_scaled = scaler.fit_transform(X)

X = pd.DataFrame(np_scaled, columns=cols)

results = {
    0.01: [],
    0.02: [],
    0.03: [],
    0.125: [],
    0.625: [],
    1: []
}
sizes = [0.01, 0.02, 0.03, 0.125, 0.625, 1]

for random_state in range(3):
    print('random_state', random_state)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state)

    for size in sizes:
        print('size', size)
        if size != 1:
            _, X_train_frac, _, Y_train_frac = train_test_split(X_train, Y_train, test_size=size, random_state=random_state)
        else:
            X_train_frac = X_train
            Y_train_frac = Y_train

        # model = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)
        # model = DecisionTreeClassifier()
        model = AdaBoostClassifier(n_estimators=1000)
        model.fit(X_train_frac, Y_train_frac)
        y_pred = model.predict(X_test)

        error = 1 - accuracy_score(Y_test, y_pred)

        results[size].append(error)

for size in sizes:
    print('average for size', size, 'equals', sum(results[size]) / (len(results[size])))

points = [sum(results[size]) / (len(results[size])) for size in sizes]

print(results)

X = np.array(sizes)
Y = np.array(points)

f, ax = plt.subplots(1)
plt.ylabel("error")
plt.xlabel("size of training data")
plt.scatter(X, Y)
n = [0.01, 0.02, 0.03, 0.125, 0.625, 1]
for i, txt in enumerate(n):
    plt.annotate(txt, (X[i], Y[i]))
plt.plot(X, Y)
plt.show()
