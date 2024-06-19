import numpy as np
import math 
import random

def train_test_split(dataset, test_size):
    formula = (len(dataset) * (test_size * 10)) / 10
    
    random.shuffle(dataset)
    test_rows = math.floor(formula)
    if test_rows == 0:
        test_rows = 1

    testing_dataset = dataset[:test_rows]
    training_dataset = dataset[test_rows:]

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for rows in training_dataset:
        x_train.append(rows[:-1])
        y_train.append(rows[-1])
        
    for rows in testing_dataset:
        x_test.append(rows[:-1])
        y_test.append(rows[-1])

    return (x_train,x_test,y_train,y_test)
    

iris_dataset = [
    [5.1, 3.5, 1.4, 0.2, 'setosa'],
    [4.9, 3.0, 1.4, 0.2, 'setosa'],
    [4.7, 3.2, 1.3, 0.2, 'setosa'],
    [5.0, 3.6, 1.4, 0.2, 'setosa'],
    [5.4, 3.9, 1.7, 0.4, 'setosa'],
    [4.6, 3.4, 1.4, 0.3, 'setosa'],
    [5.0, 3.4, 1.5, 0.2, 'setosa'],
    [4.4, 2.9, 1.4, 0.2, 'setosa'],
    [4.9, 3.1, 1.5, 0.1, 'setosa'],
    [5.4, 3.7, 1.5, 0.2, 'setosa'],
    [6.3, 3.3, 6.0, 2.5, 'virginica'],
    [6.7, 3.3, 5.7, 2.5, 'virginica'],
    [6.7, 3.0, 5.2, 2.3, 'virginica'],
    [6.5, 3.0, 5.8, 2.2, 'virginica'],
    [7.6, 3.0, 6.6, 2.1, 'virginica'],
    [7.3, 2.9, 6.3, 1.8, 'virginica'],
    [6.8, 3.2, 5.9, 2.3, 'virginica'],
    [6.9, 3.1, 5.4, 2.1, 'virginica'],
    [6.4, 2.8, 5.6, 2.1, 'virginica'],
    [6.7, 3.0, 5.2, 2.3, 'virginica']
]

x_train,x_test,y_train,y_test = train_test_split(iris_dataset,0.2)
print(x_train,end='\n')
print(y_train,end='\n')
print(x_test,end='\n')
print(y_test,end='\n')