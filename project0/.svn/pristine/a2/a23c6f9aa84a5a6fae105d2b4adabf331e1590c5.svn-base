##########################################
##### WRITE YOUR CODE IN THIS FILE #######
##########################################
import os
from load_data import load_data
from sklearn.cluster import KMeans

class GraspClustering:
    def train(self):
        training_data_path = os.environ['TRAIN_DATA_PATH']
        training_data = load_data(training_data_path)
        self.model = KMeans(n_clusters = 6, random_state = 0).fit(training_data)
        
         
    def predict(self, test_data):   	  
        predictions = self.model.predict(test_data)
        return(predictions)
