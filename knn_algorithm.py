import numpy as np
import matplotlib.pyplot as plt

def knn(dataset,target):
    coordinates = []
    X = []
    Y = []
    distance = []

    for rows in dataset:
        coordinates.append(rows[:2])
        X.append(rows[0])
        Y.append(rows[1])

    coordinates = np.array(coordinates)
    
    for coords in coordinates:
        distance.append(np.sqrt(((target[0] - coords[0]) ** 2) + ((target[1] - coords[1]) ** 2)))

    plt.title("KNN algorithm")
    plt.xlabel("X co-ordinates")
    plt.ylabel("Y co-ordinates")
    plt.scatter(X,Y,c='green',marker='x')
    plt.scatter(target[0],target[1],marker='D',c='red')
    plt.grid(True,which='both', linestyle='--', linewidth=0.5)

    # Draw lines between 'target' to all data points
    # for x, y in zip(X, Y):
    #     plt.plot([x,target[0]],[y,target[1]])

    # Draw line between target point to nearest point
    plt.plot([X[distance.index(min(distance))],target[0]],[Y[distance.index(min(distance))],target[1]])

    plt.show()

dataset = [
    [8.0,160,'A'],
    [6.2,170,'A'],
    [7.2,168,'B'],
    [8.2,155,'B']
]
testset = np.array([7.4,114])

knn(dataset,testset)