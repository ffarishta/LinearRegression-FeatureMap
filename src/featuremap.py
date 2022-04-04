import util
import numpy as np
import matplotlib.pyplot as plt
import random

np.seterr(all='raise')

factor = 2.0


class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        self.theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        # *** END CODE HERE ***

    def fit_GD(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the gradient descent algorithm.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        a = 0.01
        iterations = 10000
        self.theta = np.ones(shape =(np.shape(X[0])[0]))
        
        for i in range(iterations):
            # find the hypothesis of each row 
            pred = np.sum(self.theta*X,axis = 1)
            #calculate the cost 
            cost = pred - y 
            #take the sum of all the derivatives 
            dtheta = np.sum(X.T*cost, axis = 1)
            # plug into update rule
            self.theta = self.theta - a*(dtheta)
        # *** END CODE HERE ***
    
    def fit_SGD(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the stochastic gradient descent algorithm.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        a = 0.01
        iterations = 10000
        self.theta = np.ones(shape =(np.shape(X[0])[0]))

        m = len(y)
        for i in range(iterations):
            for i in range(m):
                # retrieve a random training input 
                rand = np.random.randint(0, m)
                pred = sum(self.theta*X[rand,:])
                cost = pred - y
                dtheta = X.T*cost
                self.theta = self.theta - a*(dtheta[:,rand])
        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        a = np.ones(shape=(X.shape[0],k+1))
        for i in range(0,k+1):
            a[:,[i]] = X[:,[1]]**i
        return a
        # *** END CODE HERE ***

    def create_cosine(self, k, X):
        """
        Generates a cosine with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        a = np.ones(shape=(X.shape[0],k+2))
        for i in range(0,k+1):
            a[:,[i]] = X[:,[1]]**i
        a[:,[k+1]] = np.cos(X[:,[1]])
        return a
        # *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return X*self.theta
        # *** END CODE HERE ***

def run_exp(train_path, cosine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.pdf'):
    
    train_x, train_y = util.load_dataset(train_path, add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-0.1, 1.1, 1000)
    plt.figure()
    plt.scatter(train_x[:, 1], train_y)
    
    for k in ks:
        
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        
        # *** START CODE HERE ***
        
        # Instantiate the linear model class
        linear_model = LinearModel()
        
        #tranform training x values into the x polynomial vector
        trans_X = linear_model.create_poly(k,train_x)
        
        #find theta 
        linear_model.fit_GD(trans_X,train_y)
        #linear_model.fit_SGD(a,train_y)
        #linear_model.fit(trans_X,train_y)

        transformed = linear_model.create_poly(k,plot_x)
        c = linear_model.predict(transformed)
        plot_y = np.sum(c,axis=1)
        
        #f_type = "Stochastic Gradient Descent"
        #f_type = "Gradient Descent"
        f_type = "Normal"
        # *** END CODE HERE ***
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
     
        plt.ylim(-2.5, 2.5)
        plt.plot(plot_x[:, 1], plot_y,
                 label='k={:d}, fit={:s}'.format(k, f_type))
        
        
        

    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()
    


def main(medium_path, small_path):
    '''
    Run all expetriments
    '''
    # *** START CODE HERE ***
    run_exp(medium_path, cosine=False, ks=[1,3, 5, 10, 20], filename='plot.pdf')
    # *** END CODE HERE ***


if __name__ == '__main__':
    main(medium_path='medium.csv',
         small_path='small.csv')
