import numpy as np
from linearregression import linearregression as LR
from logisticregession import logisticregression as Lgr
from lineardiscriminantanalysis import lineardiscriminantananlysis as Lda
from tanhiperboliqueregression import tanhiperboliqueregression as Thr
from softmaxregression import softmaxregression as sftmax


class artificialneuralnetwork_classifier :

  
  def __init__ (self,training_data_X, training_data_Y) :

    def apply_classification ():
      k = 0
      for i in self.training_data_X :
        for j in i :
          if k == 0 :
            zi = np.matrix([[ THR.predict(j), LGR.predict(j) ]])
          else :
            zi = np.insert(zi , 0 , np.matrix([[THR.predict(j), LGR.predict(j)  ]]) , axis=0)
          k += 1
      zi = np.flip(zi,0)
      return LDA = Lda(zi,self.training_data_Y)
      
          
    
    pading = np.ones(training_data_X.shape[0])
    #self.training_data_X = np.insert(training_data_X, 0, pading, axis=1) # The training data x => features numpy_matrix
    self.training_data_X = training_data_X 
    self.training_data_Y = training_data_Y # The training data y => response numpy_matrix
    Lr = LR(self.training_data_X,self.training_data_Y)
    alpha_1, rss = Lr.leastsquare()
    try :
      LGR = Lgr(self.training_data_X,self.training_data_Y)
      alpha_2 = LGR.Beta
    except :
      alpha_2 = np.zeros((alpha_1.shape[0],alpha_1.shape[1] ) )
      pass
    try :
      THR = Thr(self.training_data_X,self.training_data_Y)
      alpha_3 = THR.Beta
    except :
      alpha_3 = np.zeros((alpha_1.shape[0],alpha_1.shape[1] ) )
      pass
    try :
      SFTMAX = sftmax(self.training_data_X,self.training_data_Y)
      alpha_4 = SFTMAX.Beta
    except :
      alpha_4 = np.zeros((alpha_1.shape[0],alpha_1.shape[1] ) )
      pass
    LDA = apply_classification ()
    self.THR = THR
    self.LGR = LGR
    self.LDA = LDA
  
  def predict(self,x):
    zi = np.matrix([[ self.THR.predict(x), self.LGR.predict(x) ]])
    return LDA.predict(zi)
 
    
    

if __name__ == "__main__" :
  x = np.matrix([[1,3],[2,4],[4,1],[3,1],[4,2] ])
  y = np.matrix([[0],[0],[0],[1],[1] ] )
  Ann = artificialneuralnetwork_classifier(x,y)
  print(Ann.predict(x))
