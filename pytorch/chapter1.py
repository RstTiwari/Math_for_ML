import torch

machine = torch.tensor([2.0 , 35.0,10.0])
print(machine)  
print(machine.size)
print(machine.shape)
print(machine.ndim )


machines = torch.tensor([
    [2.9,35,10.],
    [4.2,38,13],
     [8.0, 75.0, 25.0]
])


print(machines.size)
print(machines.shape)


x = torch.tensor([[2.9,37 ,10.0]])

zero = torch.zeros(2,3)

one = torch.ones(2,3)


machines = torch.tensor([
    [2.9,35,10.],
    [4.2,38,13],
     [8.0, 75.0, 25.0]   
])

W = torch.tensor([
    [0.8,  0.2],
    [0.3, -0.4],
    [0.1,  0.7]
])


print(machines[0])
print(machines[1,2])
print(machines[:,0])
print(machines[0:2,1:2])


b = torch.tensor([0.1])

z = machines @ W + b
print(z,"the z output of the linear")

print(z.device)
print(torch.cuda.is_available())

device = "cuda" if torch.cuda.is_available() else "cpu"
z = machines @ W + b

b = torch.tensor([0.1],device=device)
print(z,"the z output of the linear")

print(b.device)
