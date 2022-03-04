import numpy as np


GLOVE_INDICES = (8, 23)


def load_data(file):
	data = np.genfromtxt(fname=file, delimiter = ',', skip_header=1)
	return data[:, GLOVE_INDICES[0]:GLOVE_INDICES[1]]
