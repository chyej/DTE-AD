import pickle
import numpy as np

with open('data/SWaT/SWaT/SWaT_train.pkl', 'rb') as f:
    data = pickle.load(f)

numpy_array = np.array(data)
np.save('data/SWaT/SWaT/SWaT_train.npy', numpy_array)