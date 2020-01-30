import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import time
from numpy.linalg import matrix_power

data = pd.read_csv("ex1data1.txt", header=None)

X = np.array((data[0]))
y = np.array((data[1]))
m = len(X)

plt.scatter(X,y)
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000")

x_0 = np.ones(m)
X = np.array([x_0, X])
theta = np.array([np.zeros(2)])

iteration = 1500
alpha = 0.01

def computeCost(X, y, theta): 
    m = len(y)
    
    
    j = 0 
    j = (1/(2*m)) * np.sum((np.square(theta.dot(X) - y)))
    
    return j

def gradientDescent(X, y, theta, iteration, alpha): 
    
    m = len(y)
    J_history = []
    for x in range(iteration):
    
        temp1 = theta[0][0] - alpha*1/m*np.sum((np.dot(theta, X) - y)*X[0])
        temp2 = theta[0][1] - alpha*1/m*np.sum((np.dot(theta, X) - y)*X[1])
        theta[0][0] = temp1
        theta[0][1] = temp2
        
        J_history.append(computeCost(X,y, theta))
    
    return theta, J_history

res1, res2 = gradientDescent(X,y,theta,iteration,alpha)

fig, ax = plt.subplots(figsize=(12,8))

fig.add_subplot()
plt.scatter(X[1], y)
plt.plot(X[1], theta.dot(X)[0], color='r')

fig, ax = plt.subplots(figsize=(12,8))
plt.plot(range(iteration), res2)

### Housing Prediction ### 

# h(theta) = -3.87805118 + 1.1912525 * x 

# 1. we want to predict if the city population is 35000 

predict = np.sum(theta.dot(np.array([[1],[17.6]]))) * 10000
print("Profit : " + str(predict) + "$" ) 

fig, ax = plt.subplots(figsize=(12,8))
plt.scatter(X[1], y)
plt.plot(X[1], theta.dot(X)[0], color='r')
plt.scatter(17.6,predict/10000)
