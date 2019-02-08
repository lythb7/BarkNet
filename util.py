import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab
import scipy.io.wavfile
import skimage.transform

import os

def processFile(path, tp):

    Fs = 44100

    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    dataList = []

    if tp == "barks":
        labelList = np.ones([len(files), 1])
    elif tp == "noise":
        labelList = np.zeros([len(files), 1])


    for file in files:
        audio = scipy.io.wavfile.read(os.path.join(path, file))
        dataList.append(audio[1])

    spec_list = np.zeros([len(files), 1024])

    for idx, data in enumerate(dataList):
        spec_mod = getSpect(data, Fs)
        spec_list[idx, :] = spec_mod
    
    final_data = np.concatenate((spec_list, labelList), axis=1)

    return final_data
    
def getData(paths_file):
    data = []
    with open(paths_file, 'r') as pfile:
        for line in pfile:
            if line == "\n":
                break
            path = line[0:-3]
            tp = "barks" if line[-2] == "b" else "noise"
            data.append(processFile(path,tp))
    data = np.concatenate(data)

    data = np.random.permutation(data)

    return data

def getSpect(data, Fs):
    spec = matplotlib.mlab.specgram(data, Fs = Fs)[0]
    spec_mod = spec[0:60, :]
    spec_mod = skimage.transform.resize(spec_mod, [32,32])
    spec_mod = np.reshape(spec_mod, [1,1024])

    return spec_mod