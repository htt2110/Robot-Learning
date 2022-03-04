
import os

import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score

from cluster_grasps import GraspClustering


GLOVE_INDICES = (9,24)


def score():
    test_data_path = os.environ['TEST_DATA_PATH']
    test_data = np.genfromtxt(fname=test_data_path, delimiter = ',', skip_header=1)
    true_labels = test_data[:, 0]

    print("Loaded: " + str(true_labels.shape[0]) + " labels")

    # Run student's scipts for training adn predicting labels
    gc = GraspClustering()
    gc.train()
    glove_data = test_data[:, GLOVE_INDICES[0]: GLOVE_INDICES[1]]
    pred_labels = gc.predict(glove_data)

    # Scoring
    score = 0
    if pred_labels.shape != true_labels.shape:
        print('Computed Score: 0 (Dimension mixmatch between the predicted labels {} and the ground truth labels {}.)'.format(pred_labels.shape, true_labels.shape))
        print('FAIL')
    else:
        score = adjusted_rand_score(true_labels, pred_labels)
        print('Computed score: ' + str(score))
        if (score > 0.72):
            print("PASS")
        else:
            print("FAIL")
    return score


if __name__ == '__main__':
    score()
