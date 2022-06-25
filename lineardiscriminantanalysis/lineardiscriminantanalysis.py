import numpy as np

class lineardiscriminantananlysis :
  
  

  def __init__(self,training_data_X, training_data_Y) :
    def get_priorprobability():
      P_y_eq_k = []
      for x in self.class_ :
        for y in x :
          p_y_eq_k = [np.sum(self.training_data_Y == y) / len(self.training_data_Y)]
          P_y_eq_k.append(p_y_eq_k)
      return P_y_eq_k
  
    def get_classspecificmeanvector():
      count = 0
      for x in self.class_ :
        for y in x :
          id = []
          c_ = 0
          for z in self.training_data_Y :
            if z == y :
              i = 1
              c_ += 1
            else :
              i = 0
            id.append(i)
          if count == 0 :
            X_i = np.matmul( np.matrix(id) , self.training_data_X)
            s = 1/c_
            classspecificmeanvector = np.matmul( np.matrix(id).dot(s) , self.training_data_X)
            count += 1
          else :
            classspecificmeanvector = np.insert(classspecificmeanvector,1 ,  np.matmul(np.matrix(id).dot(1/c_) , self.training_data_X), axis=0)
            X_i = np.insert( X_i , 1 , np.matmul( np.matrix(id) , self.training_data_X) , axis=0)
      return classspecificmeanvector , X_i

    def get_cov():
      cov = (self.X_i - self.classspecificmeanvector) * (self.X_i - self.classspecificmeanvector).T
      return cov
      
    
    # Linear regression module init
    self.training_data_X = training_data_X # The training data x => features numpy_matrix
    self.training_data_Y = training_data_Y # The training data y => response numpy_matrix
    self.class_ = np.unique(self.training_data_Y, axis=0)
    self.prioprobability = get_priorprobability()
    self.classspecificmeanvector , self.X_i = get_classspecificmeanvector()
    self.cov = get_cov()
    

if __name__ == "__main__" :
  x = np.matrix([[1,3],[2,3],[2,4],[3,1],[3,2],[4,2]])
  y = np.matrix([[1],[1],[1],[2],[2],[2]])
  Lda = lineardiscriminantananlysis(x,y)
  print(Lda.prioprobability) 
  print(Lda.classspecificmeanvector)
  print(Lda.X_i)
  print(Lda.cov) 
