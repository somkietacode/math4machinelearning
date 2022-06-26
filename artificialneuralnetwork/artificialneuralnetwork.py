import numpy as np


class artificialneuralnetwork :
  
  def __init__ (self,,training_data_X, training_data_Y) :
    
    pading = np.ones(training_data_X.shape[0])
    self.training_data_X = np.insert(training_data_X, 0, pading, axis=1) # The training data x => features numpy_matrix
    self.training_data_Y = training_data_Y # The training data y => response numpy_matrix
    
