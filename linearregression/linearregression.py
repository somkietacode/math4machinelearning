import numpy as np
from numpy.linalg import inv

class linearregression :
  
  def __init__(self,training_data_X, training_data_Y) :
    # Linear regression module init
    pading = np.ones(training_data_X.shape[0])
    self.training_data_X = np.insert(training_data_X, 0, pading, axis=1) # The training data x => features numpy_matrix
    self.training_data_Y = training_data_Y # The training data y => response numpy_matrix
  
  def leastsquare(self):
    # Find beta parameter
    X_transpose = self.training_data_X.transpose()
    self.ajk = np.matmul(X_transpose, self.training_data_X)
    self.Hessian = 2 * self.ajk
    Beta_ = np.matmul(inv(self.ajk), X_transpose) 
    self.Beta = Beta_.dot(self.training_data_Y)
    print(self.Beta)
    f_of_X = np.matmul(self.training_data_X,self.Beta)
    self.rs = np.subtract(self.training_data_Y,f_of_X)
    print(self.rs)
    self.rss = np.square(self.rs)
    return self.Beta , self.rss
    
if __name__ == "__main__" :
  x = np.matrix([[0,1],[1,4],[7,8]])
  y = np.matrix([[2],[9],[23]])
  Lr = linearregression(x,y)
  Beta, rss = Lr.leastsquare()
  print(Beta,rss)
    
