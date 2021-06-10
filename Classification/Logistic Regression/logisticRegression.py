import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

class LogisticRegressionModal:
    """
    Logistic Regression.
    LogisticRegressionModal fits a sigmoid function based hypothesis with coefficients w = (w1, ..., wp)
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by approximation.
    
    params:
    print_details : if True, returns the details after every epoch 
    X: X_train is the training dataset with m samples (rows) and n features (columns)
    y: Y_train is the training dataset with m target values 
    alpha: alpha is the learning rate for the gradient descent steps
    threshold : default = 0.5. Threshold decides the value where the probability values assumes y = 1 
    num_epochs: it defines the number of iterations of gradient descent 
    
    returns:
    
    fit() : This method fit the X, y values and returns the values of cost function 
    after every epoch, the final thetha values of the hyothesis, 
    the mean of each column and standard deviation of each column.
    
    predict() : This method predicts the value of new sample 
    with the theta values captured in fit(). 
    y_pred = 1/ (1 + e^(-(theta0+ theta1x1 + theta2x2 + .... )))
    
    getAccuracy() : This method returns the mean squared error of the 
    logistic regression model 
    
    -------------------------
    Example : 

        from sklearn.datasets import load_boston
        X_train, y_train = load_boston(return_X_y=True)
        reg = LinearRegressionModal()
        model = reg.fit(X_train, y_train, alpha=0.1, num_epochs=100)

        X_test = X_train[:10 , :]
        reg.predict(X_test, model)
            
    """
    
    def __init__(self, print_details = False):
        self.mean_array = []
        self.std_array = []
        self.print_details = print_details

     
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
        z = np.dot(X, theta)
        H = 1/(1 + (np.exp(-z))) + 0.000001
        return H
    
    def cost(self, X, theta, y ):
        term1 = - y@np.log(self.h(X,theta))
        term2 = - (1-y)@np.log(1 - self.h(X,theta))
        
        return term1+ term2
    
    def J(self, X, theta, y):
        
        cost = self.cost(X, theta,y)
        
        j = (1/X.shape[0])* (cost)
        return j
    
    
    
    def gradientDescent(self, X, y, alpha, num_epochs):
        J_hist = []
        theta = self.thetas(X, 0.0)
        for epoch in range(num_epochs):
            H = self.h(X, theta)
            J_epoch = self.J(X, theta, y)
            J_hist.append(J_epoch)
            if self.print_details:
                print("epoch " , epoch , " -------------> J(theta) = ", J_epoch , "\n")
            descent = alpha*( (1/X.shape[0])* np.dot( (X.T), H - y )  )
            theta = theta - descent

        return J_hist, theta
    
    def accuracy(self,y_pred, y):
        root_mean_squared_error = np.sqrt(np.mean((y_pred - y).T@(y_pred - y)))
        return root_mean_squared_error
    
    def predict(self, X_test):
        X_test_norm = self.normaliseTest(X_test, self.model[2], self.model[3])
        X_test = self.addNewColumn(X_test_norm)
        prediction = self.h(X_test, self.model[1])
        prediction[prediction>=self.threshold] = 1
        prediction[prediction<self.threshold] = 0
        return prediction
        
    def fit(self, X, y, alpha, threshold, num_epochs ):
        #feature scaling 
        X, mean, std =  self.normalise(X)
        # add new column of 1's to X
        X = self.addNewColumn(X)
        # Returns the cost function and the final theta values
        J_values, params= self.gradientDescent(X, y, alpha, num_epochs)
        
        #find accuracy
        y_pred = self.h(X, params)
        
        #applying threshold 
        self.threshold = threshold
        
        y_pred[y_pred >= threshold] = 1
        y_pred[y_pred < threshold] = 0
        
        self.y_pred = y_pred 
        self.y = y 
        mse = self.accuracy(self.y_pred, self.y)
        
        print("MSE ---->" , mse)
        if self.print_details:
            print('Final Cost function value ---->' , J_values[-1], "\n")
            print('Parameters of the linear regression ---->',"\n", str(params), "\n")
            

            plt.plot(J_values)
            plt.title('Cost function for alpha = '+ str(alpha))
            plt.xlabel('Iterations')
            plt.ylabel('Cost function')

        self.model = [J_values, params, mean, std, mse]
        return self.model
    
    
    def getAccuracy(self):
        return self.accuracy(self.y_pred, self.y)

