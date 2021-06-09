import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

class LinearRegressionModal:
    """
    Ordinary least squares Linear Regression.
    LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation.
    
    -----
    Examples
    --------
    >>> import numpy as np
    >>> import LinearRegressionModal
    >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    >>> # y = 1 * x_0 + 2 * x_1 + 3
    >>> y = np.dot(X, np.array([1, 2])) + 3
    >>> reg = LinearRegressionModal().fit(X, y, alpha=0.01, num_epochs = 100)
    >>> predict(np.array([[3, 5]]), reg)
    array([16.])
    """
    def __init__(self):
        self.mean_array = []
        self.std_array = []

     
    def normalise(self, X):
        for i in range(X.shape[1]):
            mean = X[:,i].mean()
            self.mean_array.append(mean)

            std = X[:,i].std()
            self.std_array.append(std)

            X[:, i] = (X[:,i] - mean)/ (std)


        return X,  self.mean_array, self.std_array   
            
    def normaliseTest(self, X, mean_array, std_array ):
        for i in range(X.shape[1] -1 ):
            X[:,i]= (X[:,i]- mean_array[i])/(std_array[i])

        return X
    
    def addNewColumn(self, X):
    
        new_column = np.array([1]*X.shape[0])
        X = np.insert(X, 0, new_column,axis=1)
        return X
    
    def thetas(self, X, initial_value):
        return np.array([initial_value]*X.shape[1])

    def h(self, X, theta):
        return X@theta
    
    def J(self, X, theta, y):
        return (((self.h(X, theta) - y).T@(self.h(X,theta)-y)))/(2*X.shape[0])
    
    def gradientDescent(self, X, y, alpha, num_epochs):
        J_hist = []
        theta = self.thetas(X, 0.0)
        for epoch in range(num_epochs):
            H = self.h(X, theta)
            J_epoch = self.J(X, theta, y)
            J_hist.append(J_epoch)
            print("epoch " , epoch , " -------------> J(theta) = ", J_epoch , "\n")
            descent = alpha*( (1/X.shape[0])* np.dot( (X.T), H - y )  )
            theta = theta - descent

        return J_hist, theta
    
    def accuracy(self,y_pred, y):
        root_mean_squared_error = np.sqrt(np.mean((y_pred - y).T@(y_pred - y)))
        print("MSE ----------->" , root_mean_squared_error)   
    
    def predict(self, X_test, model):
        X_test_norm = self.normaliseTest(X_test, model[2], model[3])
        X_test = self.addNewColumn(X_test_norm)
        return np.dot(X_test, model[1])
        
    def fit(self, X, y, alpha, num_epochs ):
        #feature scaling 
        X, mean, std =  self.normalise(X)
        # add new column of 1's to X
        X = self.addNewColumn(X)
        # Returns the cost function and the final theta values
        J_values, params= self.gradientDescent(X, y, alpha, num_epochs)
        
        #find accuracy
        y_pred = np.dot(X,params)
        print('Final Cost function value ---->' , J_values[-1], "\n")
        print('Parameters of the linear regression ---->',"\n", str(params), "\n")
        mse = self.accuracy(y_pred, y)

        plt.plot(J_values)
        plt.title('Cost function for alpha = '+ str(alpha))
        plt.xlabel('Iterations')
        plt.ylabel('Cost function')

        
        return J_values, params, mean, std


from sklearn.datasets import load_boston
X_train, y_train = load_boston(return_X_y=True)
model = LinearRegressionModal().fit(X_train, y_train, alpha=0.1, num_epochs=100)

X_test = X_train[:10 , :]
LinearRegressionModal().predict(X_test, model)

y_train[:10]