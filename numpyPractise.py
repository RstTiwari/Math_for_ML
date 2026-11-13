import numpy as np

vibration = np.array([2.1, 3.5, 4.2, 7.8, 8.1])

print(vibration)
print(vibration.shape)
print(vibration.ndim)
print(vibration.size)
print(vibration.dtype)

data = np.array([
    [2.1, 35],
    [3.5, 38],
    [4.2, 40],
    [7.8, 75],
    [8.1, 80]])

print(data)
print(data.shape)
print(data.ndim)
print(data.size)


#indexing

print(data[0])
print(data[0][1])
print(data[3][1])

#slicing
print(data[:,0])
print(data[:,:])
print (data[0:2,0:])


#let perform mathematical opretion on

print(vibration+2)
print(vibration*2)
print(vibration**2)
print(np.mean(vibration))
print(np.median(vibration))

readings = np.array([10, 10, 10, 10])

print(np.mean(readings))
print(np.var(readings))



print(data.T)
print(data.T.shape)

#matrix matiplication
inputs = np.array([2.0, 35])
weights = np.array([0.8, 0.2])
# bias = 2.0
result = np.dot(inputs,weights) 
print(result,"result of matrix multiplication")

inputs1 = np.array([3, 5])
weights1 = np.array([2, 4])

result1 = np.dot(inputs1, weights1)
print(result1, "result of matrix multiplication")


import numpy as np

# Three input features
X = np.array([[0.8, 0.6, 0.2]])


# Three features connected to two neurons
W = np.array([
    [0.8, 0.2],
    [0.3, -0.4],
    [0.1, 0.7]
])

# One bias per neuron
b = np.array([[0.1, -0.05]])

# Neuron calculations
Z = np.dot(X, W) + b

print(X.shape)
print(W.shape)


#let add more input vibration and t emprature
input = np.array([
    [0.8, 0.6, 0.2],
    [0.5, 0.3, 0.9]
  ])

z = np.dot(input,W) + b
print(z)
