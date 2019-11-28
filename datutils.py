import cPickle as pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import random

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



    def loadData(self,ROOT="",num=8):

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

        print "We have %d training examples" % X_tr.shape[0]
        print "and there are %d test examples" % X_te.shape[0]
        print "over %d classes" % len(set(Y_tr))


        X_tr, Y_tr = self.shuffleTrain(X_tr,Y_tr)

        return X_tr, Y_tr, X_te, Y_te

    #randomize the data which by default is on class order
    def shuffleTrain(self, Xtr, Ytr):

        inds = random.sample(range(0,Xtr.shape[0]),Xtr.shape[0])
        shuffledX = Xtr[inds]
        shuffledY = Ytr[inds]

        return shuffledX, shuffledY

    #split data into 5 folds
    def genXvalFolds(self,Xtr,Ytr):

        Xtr,Ytr = self.shuffleTrain(Xtr,Ytr)

        #5 folds
        offset = Xtr.shape[0]//5

        Xfolds = [[]]
        Yfolds = [[]]

        Xfolds[0] = Xtr[0:offset]
        Xfolds.append(Xtr[offset:2*offset])
        Xfolds.append(Xtr[2*offset:3*offset])
        Xfolds.append(Xtr[3*offset:4*offset])
        Xfolds.append(Xtr[4*offset:])

        Yfolds[0] = Ytr[0:offset]
        Yfolds.append(Ytr[offset:2*offset])
        Yfolds.append(Ytr[2*offset:3*offset])
        Yfolds.append(Ytr[3*offset:4*offset])
        Yfolds.append(Ytr[4*offset:])

        return Xfolds, Yfolds

    #get traininf ddata composed of all folds except valFold whic we use for the validation set
    def getTrainVal(self,Xtr,Ytr,valFold):
        Xfolds,Yfolds = self.genXvalFolds(Xtr,Ytr)
        Xtrain=np.empty((0,Xfolds[0].shape[1],Xfolds[0].shape[2],Xfolds[0].shape[3]))
        Ytrain=np.array([])
        Xval = np.array(Xfolds[valFold])
        Yval = np.array(Yfolds[valFold])

        for i in range(0,5):
            #print "Xfold %d shape is:" % i
            #print Xfolds[i].shape
            if i!=valFold:
                #print "i!=valFold:"
                #print "Before append, XTrai shape is:"
                #print Xtrain.shape
                Xtrain = np.append(Xtrain,np.array(Xfolds[i]),axis=0)
                Ytrain = np.append(Ytrain,np.array(Yfolds[i]),axis=0)
                #print "After append, Xtrain shape is:"
                #print Xtrain.shape
                
        #print "After amalgamamating folds:"
        #print "Xtr shape is:"
        #print Xtrain.shape
        #print "Ytrain shape is:"
        #print Ytrain.shape
        #print Ytrain
         

        return Xtrain, Ytrain, Xval, Yval


    def flattenData(self,Xtr,Xte, Xval):
        #flatten out all images to be one-dimensional
#       Xtr_rows = Xtr.reshape(Xtr.shape[0],1935*2592*3)
#        Xte_rows = Xte.reshape(Xte.shape[0],1935*2592*3)
        Xtr_rows = Xtr.reshape(Xtr.shape[0],Xtr.shape[1]*Xtr.shape[2]*Xtr.shape[3])
        Xte_rows = Xte.reshape(Xte.shape[0],Xte.shape[1]*Xte.shape[2]*Xte.shape[3])
        Xval_rows = Xval.reshape(Xval.shape[0],Xval.shape[1]*Xval.shape[2]*Xval.shape[3])

        return Xtr_rows, Xte_rows, Xval_rows

    def addBias(self,Xtr,Xte,Xval):
        #append ones to each column of the image data
        Xtrbias = np.ones((Xtr.shape[0],1))
        Xtebias = np.ones((Xte.shape[0],1))
        Xvalbias = np.ones((Xval.shape[0],1))

        #print Xtrbias
        #print Xtebias

        Xtrstack = np.hstack((Xtr,Xtrbias))
        Xtestack = np.hstack((Xte,Xtebias))
        Xvalstack = np.hstack((Xval,Xvalbias))

        return Xtrstack, Xtestack, Xvalstack
        

    def getMeanImg(self,Xtr):
        meanImg = np.mean(Xtr, axis=0)
        return meanImg

    def showMeanImg(self,Xtr):
        plt.figure(figsize=(4,4))
        plt.imshow(self.getMeanImg(Xtr).reshape((Xtr.shape[1],Xtr.shape[2],Xtr.shape[3])).astype('uint8')) # visualize the mean image
        plt.show()

    def subMeanImg(self,Xtr,Xte,Xval):
        mean = self.getMeanImg(Xtr)
        Xtr -= mean
        Xte -= mean
        Xval -= mean
        return Xtr, Xte, Xval


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




