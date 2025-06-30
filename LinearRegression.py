import numpy as np 

class LinearRegression:
    def __init__(self,learning_rate=0.01,epochs=1000):
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.weights=None
        self.bias=None

    def fit(self,X,y):
        n_samples,n_features=X.shape
        self.weights=np.zeros(n_features)
        self.bias=0

        #Gradient Descent
        for _ in range(self.epochs):
            y_pred=np.dot(X,self.weights) + self.bias #Linear model prediction

            #Compute gradients
            dw=(1/n_samples)*np.dot(X.T,(y_pred-y))
            db=(1/n_samples)*np.sum(y_pred-y)

            #Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self,X):
        return np.dot(X,self.weights) + self.bias
    
#Example
if __name__=="__main__":
    #Input Data (X) and Output LAbel (Y)
    X=np.array([[1],[2],[3],[4],[5]])
    y=np.array([1,2,3,4,5])

    #Create Linear Regression Model
    model=LinearRegression(learning_rate=0.01,epochs=1000)
    model.fit(X,y)

    #Predict new values
    prediction = model.predict(X)
    print("Predictions:",prediction)


