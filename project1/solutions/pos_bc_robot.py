from base import RobotPolicy
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression

class POSBCRobot(RobotPolicy):
    
    """ Implement solution for Part 2 below """

    def train(self, data):
        #for key, val in data.items():
        #    print(key, val.shape)
        #print("Using dummy solution for POSBCRobot")
        #pass
        X = data["obs"]
        y = np.array(data["actions"])
        y=np.ravel(y)
        self.clf = Perceptron(penalty = 'l2', fit_intercept = True, tol=1e-4, class_weight = "balanced", random_state=0).fit(X,y) 
        #self.clf = LogisticRegression(penalty='l2', tol=0.0001, C=5.0, random_state=0, solver='newton-cg', multi_class='multinomial').fit(X,y)                                                                                     

    def get_actions(self, observations):
        preds = self.clf.predict(observations)
        #preds_2 = self.clf2.predict(observations)
        #preds = (preds_1 + preds_2)//2
        return preds #np.zeros(observations.shape[0])
