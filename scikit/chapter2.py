import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X = np.array([
      [2, 30],
       [3, 35],
       [4, 40],
       [5, 45],
       [6, 55],
       [7, 65],
       [8, 70],
       [9, 80],
       [2.5, 33],
       [3.5, 38],
       [4.5, 43],
       [5.5, 50],
       [6.5, 60],
       [7.5, 68],
       [8.5, 75],
       [9.5, 85],
       [3, 42],
       [4, 48],
       [7, 62],
       [8, 78]
])


y = np.array([
    0, 0, 0, 0,
    1, 1, 1, 1,
    0, 0, 0, 1,
    1, 1, 1, 1,
    0, 0, 1, 1
])


X_train,X_test,y_train,y_test = train_test_split(X ,y, test_size= 0.2, random_state = 42)

model = LogisticRegression()
model.fit(X_train,y_train)


print("print_weigh", model.coef_)
print("print_biases", model.intercept_)


y_pred = model.predict(X_test)
print("predicted value", y_pred)
print("actual value", y_test)


train_accuracy = model.score(X_train,y_train)
test_accuracy = model.score(X_test,y_test)

print("train_accuracy", train_accuracy)
print("test_accuracy", test_accuracy)   