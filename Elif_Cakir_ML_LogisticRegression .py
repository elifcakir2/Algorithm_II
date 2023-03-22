#!/usr/bin/env python
# coding: utf-8

# # Machine Learning - Andrew Ng ( Python Implementation)
# 
# ##  Logistic Regression

# ### Loading of Data

# In[90]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[91]:


df=pd.read_csv("ex2data1.txt",header=None)
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values


# In[92]:


df.head()


# In[93]:


df.describe()


# ### Plotting of Data

# In[94]:


pos , neg = (y==1).reshape(100,1) , (y==0).reshape(100,1)
plt.scatter(X[pos[:,0],0],X[pos[:,0],1],c="r",marker="+")
plt.scatter(X[neg[:,0],0],X[neg[:,0],1],marker="o",s=10)
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend(["Admitted","Not admitted"],loc=0)


# ### Sigmoid function
# 
# $ g(z) = \frac{1}{(1+e^{-z})}$

# In[95]:


def sigmoid(z):
    """
    return the sigmoid of z
    """
    
    return 1/ (1 + np.exp(-z))


# In[96]:


# testing the sigmoid function
sigmoid(0)


# ### Compute the Cost Function and Gradient
# 
# $J(\Theta) = \frac{1}{m} \sum_{i=1}^{m} [ -y^{(i)}log(h_{\Theta}(x^{(i)})) - (1 - y^{(i)})log(1 - (h_{\Theta}(x^{(i)}))]$
# 
# $ \frac{\partial J(\Theta)}{\partial \Theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_{\Theta}(x^{(i)}) - y^{(i)})x_j^{(i)}$

# In[97]:


def costFunction(theta, X, y):
    """
    Takes in numpy array theta, x and y and return the logistic regression cost function and gradient
    """
    
    m=len(y)
    
    predictions = sigmoid(np.dot(X,theta))
    error = (-y * np.log(predictions)) - ((1-y)*np.log(1-predictions))

    cost = 1/m * sum(error)
    
    grad = 1/m * np.dot(X.transpose(),(predictions - y))
    
    return cost[0] , grad


# ### Feature scaling

# In[98]:


def featureNormalization(X):
    """
    Take in numpy array of X values and return normalize X values,
    the mean and standard deviation of each feature
    """
    mean=np.mean(X,axis=0)
    std=np.std(X,axis=0)
    
    X_norm = (X - mean)/std
    
    return X_norm , mean , std


# In[99]:


m , n = X.shape[0], X.shape[1]
X, X_mean, X_std = featureNormalization(X)
X= np.append(np.ones((m,1)),X,axis=1)
y=y.reshape(m,1)
initial_theta = np.zeros((n+1,1))
cost, grad= costFunction(initial_theta,X,y)
print("Cost of initial theta is",cost)
print("Gradient at initial theta (zeros):",grad)


# ### Gradient Descent

# In[100]:


def gradientDescent(X,y,theta,alpha,num_iters):
    """
    Take in numpy array X, y and theta and update theta by taking num_iters gradient steps
    with learning rate of alpha
    
    return theta and the list of the cost of theta during each iteration
    """
    
    m=len(y)
    J_history =[]
    
    for i in range(num_iters):
        cost, grad = costFunction(theta,X,y)
        theta = theta - (alpha * grad)
        J_history.append(cost)
    
    return theta , J_history


# In[101]:


theta , J_history = gradientDescent(X,y,initial_theta,1,400)


# In[102]:


print("Theta optimized by gradient descent:",theta)
print("The cost of the optimized theta:",J_history[-1])


# ### Plotting of Cost Function

# In[103]:


plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")


# ### Plotting the decision boundary
# 
# From Machine Learning Resources:
#     
# $h_\Theta(x) = g(z)$, where g is the sigmoid function and $z = \Theta^Tx$
# 
# Since $h_\Theta(x) \geq 0.5$ is interpreted as predicting class "1", $g(\Theta^Tx) \geq 0.5$ or $\Theta^Tx \geq 0$ predict class "1" 
# 
# $\Theta_1 + \Theta_2x_2 + \Theta_3x_3 = 0$ is the decision boundary   
# 
# Since, we plot $x_2$ against $x_3$, the boundary line will be the equation $ x_3 = \frac{-(\Theta_1+\Theta_2x_2)}{\Theta_3}$

# In[104]:


plt.scatter(X[pos[:,0],1],X[pos[:,0],2],c="r",marker="+",label="Admitted")
plt.scatter(X[neg[:,0],1],X[neg[:,0],2],c="b",marker="x",label="Not admitted")
x_value= np.array([np.min(X[:,1]),np.max(X[:,1])])
y_value=-(theta[0] +theta[1]*x_value)/theta[2]
plt.plot(x_value,y_value, "g")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend(loc=0)


# ### Prediction

# In[105]:


def classifierPredict(theta,X):
    """
    take in numpy array of theta and X and predict the class 
    """
    predictions = X.dot(theta)
    
    return predictions>0


# In[106]:


x_test = np.array([45,85])
x_test = (x_test - X_mean)/X_std
x_test = np.append(np.ones(1),x_test)
prob = sigmoid(x_test.dot(theta))
print("For a student with scores 45 and 85, we predict an admission probability of",prob[0])


# ### Accuracy on training set 

# In[107]:


p=classifierPredict(theta,X)
print("Train Accuracy:", sum(p==y)[0],"%")


# In[108]:


###Train-test split 80%-20%

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 1)
logreg = LogisticRegression(solver='newton-cg', max_iter=200)
logreg.fit(X_train,y_train)


# In[ ]:


y_prediction = logreg.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_prediction)


# In[ ]:


from sklearn.metrics import precision_score
precision_score(y_test,y_prediction)

