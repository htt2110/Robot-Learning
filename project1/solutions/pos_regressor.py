from base import Regressor
import numpy as np
from sklearn import linear_model
class PositionRegressor(Regressor):

    """ Implement solution for Part 1 below  """

    def train(self, data):
        #for key, val in data.items():
        #    print(key, val)
        #print("Using dummy solution for PositionRegressor")
        #pass
        #print(data)
        X = data["obs"].reshape(500, 64*64*3)
        #print(X.shape)
        y = np.asarray([info['agent_pos'] for info in data['info']])
        self.reg = linear_model.Ridge(alpha=.5).fit(X, y)

    def predict(self, Xs):
        #print(Xs.shape)
        Xs_adj = Xs.reshape(Xs.shape[0], 64*64*3)
        predictions = self.reg.predict(Xs_adj)
        #print(predictions.shape)
        return predictions #np.zeros((Xs.shape[0], 2))
