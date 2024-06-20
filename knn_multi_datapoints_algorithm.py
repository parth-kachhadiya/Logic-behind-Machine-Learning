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
    
    for cods in target:
        ls = []
        for vl in coordinates:
            ls.append(np.sqrt(((vl[0] - cods[0]) ** 2) + ((vl[1] - cods[1]) ** 2)))    
        distance.append(ls)


    plt.title("KNN algorithm")
    plt.xlabel("X co-ordinates")
    plt.ylabel("Y co-ordinates")
    plt.scatter(X,Y,c='green',marker='X',zorder=4,s=150)
    
    for tg in target:
        plt.scatter(tg[0],tg[1],marker='D',c='red',zorder=4)
    
    plt.grid(True,which='both', linestyle='--', linewidth=0.5)

    for r_1,r_3 in zip(distance,target):
        plt.plot([X[r_1.index(min(r_1))],r_3[0]],[Y[r_1.index(min(r_1))],r_3[1]],c='orange')
 
    plt.show()

dataset = [
    [8.0,160,'A'],
    [6.2,170,'A'],
    [7.2,168,'B'],
    [8.2,155,'B']
]
testset = [[7.4,144],
    [6.1,150],
    [9.3,164],
    [4.4,180],
]

knn(dataset,testset)
