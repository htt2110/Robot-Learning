from base import RobotPolicy
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

class RGBBCRobot(RobotPolicy):

    """ Implement solution for Part3 below """

    def train(self, data):
        #for key, val in data.items():
        #    print(key, val.shape)
        #print("Using dummy solution for RGBBCRobot")
        #pass
        y = np.array(data["actions"])
        y = np.ravel(y)
        X = data["obs"].reshape(400,64*64*3)
        self.transform = PCA(n_components=6, svd_solver = 'arpack', random_state = 24).fit(X) #random_state = 0,24,28,39 n_components = 7,9, gives 3/5
        X_pca = self.transform.transform(X)
        #self.clf = KNeighborsClassifier(n_neighbors=8).fit(X,y) #gives score 2/5 with 8 neighbors
        self.clf = LogisticRegression(penalty = 'l2', C= 5, multi_class='multinomial', max_iter = 1000, tol = 5e-2, solver = 'newton-cg', random_state = 24).fit(X_pca,y) 
                                      

    def get_actions(self, observations):
       #obs_centered = self.scaler.transform(observations)
       obs_transform = self.transform.transform(observations)   
       preds = self.clf.predict(obs_transform)
       return preds #np.zeros(observations.shape[0])
