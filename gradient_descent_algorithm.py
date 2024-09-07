import numpy as np
import matplotlib.pyplot as plt

def gredient_descent(x, y):
    """
    NOTE: Based on the 'iterations' and 'learning_rate' values, model performance will change and
          The rate of change will be displayed on plot
    """
    # Step - 1 : Initialize the value of 'm' and 'b' as random value
    M, B = 0, 0
    n = len(x)
    plt_y_hat = None
    
    # Step - 2 : Number of iterations 
    iterations = 100

    # Initialize 'learning_rate' as random value, do more Trial and error and set the optimal value of 'learning_rate'
    learning_rate = 0.01 

    # Scatter plot of actual values
    plt.scatter(x, y,color="Red",marker="+",s=100,linewidths=3)
    plt.xlabel("Random values of X")
    plt.ylabel("Random values of Y")
    plt.title("Gradient descent algorithm test")

    for i in range(iterations):
        y_hat = M * x + B
        plt_y_hat = y_hat

        MSE = (1/n) * sum([val ** 2 for val in (y - y_hat)])
        
        partialDerivativeOf_m = -(2/n) * sum(x * (y - y_hat))
        partialDerivativeOf_b = -(2/n) * sum(y - y_hat)

        M = M - learning_rate * partialDerivativeOf_m
        B = B - learning_rate * partialDerivativeOf_b

        print(f"iteration : {i}, m : {M}, b : {B}, MSE : {MSE}")
    
    plt.plot(x,plt_y_hat,color='green')
    plt.show()

X = np.array([1,2,3,4,5])
Y = np.array([5,7,9,11,13])

gredient_descent(X,Y)
