import numpy as np

def euclidian_dis(X,Y):
    '''
    All parametters must be type of 'np.ndarray'
    '''
    if isinstance(X,np.ndarray) and isinstance(Y,np.ndarray):
        if len(X) == len(Y):
            distance = np.sqrt(sum((Y - X) ** 2))
            return distance
        else:
            return -1
    else:
        return -1

x = np.array([1,2,3])
y = np.array([4,5,6])

print(euclidian_dis.__doc__)
print(euclidian_dis(x,y))