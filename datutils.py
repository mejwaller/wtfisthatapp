import cPickle as pickle
import os
import numpy as np

def loadTraining(ROOT,num):

    xs = []
    ys = []
    for i in range(1,num+1):
        trainfname = os.path.join(ROOT,'train%d.pkl' % (i,))
        labelsfname = os.path.join(ROOT,'indexes%d.pkl' % (i,))
        print "Loading " + trainfname
        with open(trainfname,'rb') as f:
            X = pickle.load(f)
            xs.append(X)

        print "Loading " + labelsfname
        with open(labelsfname,'rb') as  f:
            Y = pickle.load(f)
            ys.append(Y)

    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)

    return Xtr,Ytr



def loadData(ROOT="",num=3):

    X_tr, Y_tr = loadTraining(ROOT,num)

    filename=ROOT+"test.pkl"

    print "Loading " + filename
    with open(filename,'rb') as f:
        X_te = pickle.load(f)

    X_te = np.array(X_te)
    
    filename=ROOT+"testinds.pkl"
    
    print "Loading " + filename
    with open(filename,'rb') as f:
        Y_te = pickle.load(f)
    
    Y_te = np.array(Y_te)

    return X_tr, Y_tr, X_te, Y_te

def flattenData(Xtr,Xte):
    #flatten out all images to be one-dimensional
#    Xtr_rows = Xtr.reshape(Xtr.shape[0],1935*2592*3)
#    Xte_rows = Xte.reshape(Xte.shape[0],1935*2592*3)
    Xtr_rows = Xtr.reshape(Xtr.shape[0],Xtr.shape[1]*Xtr.shape[2]*Xtr.shape[3])
    Xte_rows = Xte.reshape(Xte.shape[0],Xte.shape[1]*Xte.shape[2]*Xte.shape[3])

    return Xtr_rows, Xte_rows

def biasTrick(X_tr,X_te):
    #append ones to each column of the image data
    pass

