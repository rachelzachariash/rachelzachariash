import pickle
from phase4.data_holder import DataHolder
from phase4.tflman import TflMan
# import TensorFlow
from tensorflow.python.keras.models import load_model
import numpy as np
class Controller:
    @staticmethod
    def load_pkl_data(pkl_path, frames):
        with open(pkl_path, 'rb') as pklfile:
            data = pickle.load(pklfile, encoding='latin1')
        focal = data['flx']
        pp = data['principle_point']
        id = int(frames[0][:-1])
        frames=frames[1:]
        EM = np.eye(4)
        for i in range(len(frames)-1):
            EM = np.dot(data['egomotion_' + str(id +i) + '-' + str(id +i + 1)], EM)
        return focal, pp, EM



    @staticmethod
    def load_pls_data(pls_path):
        with open(pls_path, "r") as file:
            data = file.readlines()
        pkl_file = data[0]
        frames = data[1:]
        return frames, pkl_file

    def run(self, pls_path):
        loaded_model = load_model('model.h5')
        frames, pkl_path = Controller.load_pls_data(pls_path)
        focal, pp, EMs = Controller.load_pkl_data(pkl_path[:-1], frames)
        dh = DataHolder(pp, focal)
        tfl_manager = TflMan()
        frames=frames[1:]
        for i, frame in enumerate(frames[:-1]):
            dh.EM = EMs
            tfl_manager.run_on_frame(frame[:-1], dh, loaded_model)
