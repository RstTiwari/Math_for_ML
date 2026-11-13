X_train = [[2], [3], [4], [7], [5], [6], [8], [9]]

y_train = [
    "Good", "Good", "Good", "Defective",
    "Good", "Defective", "Defective", "Defective"
]

X_test = [[2], [4], [5], [10], [11], [7], [13], [1]]

X_shfulle = [[8,10,9,7,3,1,11,15]]
Y_shufulle = ["Defective","Defective","Defective","Defective","Good","Good","Defective","Defective"]

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators= 100,
    max_depth=10,
    random_state=42
)

model.fit(X_train,y_train)
prediction = model.predict(X_test)
print(prediction)


# let's have permutaion importance for the feature to learn abou8 that
from sklearn.inspection import permutation_importance

result = permutation_importance(
    model,
    X_shfulle,
    Y_shufulle,
    
)
