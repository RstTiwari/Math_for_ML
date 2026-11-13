import numpyPractise as np
step = np.arange(0,8,0.1)


def velocity(t):
    return -t**2 + 8*t

def calculateTheArea(dt):
    # use numpy to create the point intervals at give points
    total_steps = np.arange(0,8,dt)
    
    #height at given point is v(t)== -t**2+8t
    height = velocity(total_steps)
    
    total_area = np.sum(height*dt)
    return total_area



print("the total area under the grap at give point 1.0 is" ,calculateTheArea(1.0))
print("the total area under the grap at give point 0.1 is" ,calculateTheArea(0.1))
print("the total area under the grap at give point 0.013 is" ,calculateTheArea(0.013))

