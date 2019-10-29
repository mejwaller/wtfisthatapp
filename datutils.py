import cPickle as pickle
import os
import numpy as np
import matplotlib.pyplot as plt

class datutils:

    def __INIT__(self):
        pass

    def loadTraining(self,ROOT,num):

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



    def loadData(self,ROOT="",num=6):

        X_tr, Y_tr = self.loadTraining(ROOT,num)

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

    def flattenData(self,Xtr,Xte):
        #flatten out all images to be one-dimensional
#       Xtr_rows = Xtr.reshape(Xtr.shape[0],1935*2592*3)
#        Xte_rows = Xte.reshape(Xte.shape[0],1935*2592*3)
        Xtr_rows = Xtr.reshape(Xtr.shape[0],Xtr.shape[1]*Xtr.shape[2]*Xtr.shape[3])
        Xte_rows = Xte.reshape(Xte.shape[0],Xte.shape[1]*Xte.shape[2]*Xte.shape[3])

        return Xtr_rows, Xte_rows

    def addBias(self,Xtr,Xte):
        #append ones to each column of the image data
        Xtrbias = np.ones((Xtr.shape[0],1))
        Xtebias = np.ones((Xte.shape[0],1))

        #print Xtrbias
        #print Xtebias

        Xtrstack = np.hstack((Xtr,Xtrbias))
        Xtestack = np.hstack((Xte,Xtebias))

        return Xtrstack, Xtestack
        

    def getMeanImg(self,Xtr):
        meanImg = np.mean(Xtr, axis=0)
        return meanImg

    def showMeanImg(self,Xtr):
        plt.figure(figsize=(4,4))
        plt.imshow(self.getMeanImg(Xtr).reshape((Xtr.shape[1],Xtr.shape[2],Xtr.shape[3])).astype('uint8')) # visualize the mean image
        plt.show()

    def subMeanImg(self,Xtr,Xte):
        mean = self.getMeanImg(Xtr)
        Xtr -= mean
        Xte -= mean
        return Xtr, Xte


'''
    def visualise(self):
        Xtr,Ytr,Xte,Yte = self.loadData()
        num_classes = len(set(Ytr))
        samples_per_class = 3

        for y, cls in enumerate(np.arange(num_classes)):
            idxs = np.flatnonzero(Ytr==y)
            idxs = np.random.choice(idxs, samples_per_class, replace=False) #ValueError: Cannot take a larger sample than population when 'replace=False'
            for i, idx in enumerate(idxs):
                plt_idx=i*num_classes+y+1
                plt.subplot(samples_per_class, num_classes, plt_idx)
                plt.imshow(Xtr[idx].astype('uint8'))
                plt.axis("off")
         plt.show()
'''




