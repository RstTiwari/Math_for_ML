import numpy as np

X = np.array([
    [2, 30],
    [3, 35],
    [4, 40],
    [5, 45],
    [6, 55],
    [7, 65],
    [8, 70],
    [9, 80]
])


y = np.array([0, 0, 0, 0, 1, 1, 1, 1])


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
print(model)

model.fit(X,y)


print("model cpfee", model.coef_)
print("intrecept", model.intercept_)

new_machine = np.array([
    [2.56,31],
    [9.32, 75]
])

prediction = model.predict(new_machine)
print("prediction", prediction)


print("probabilty", model.predict_proba(new_machine))