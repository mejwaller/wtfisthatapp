import datutils as dat
import random
import SVMLoss
import numpy as np

#hyperparameters: reg, step size
#use 4 epochs for corase search!

#9 examples of each, randomized

du = dat.datutils()

Xtr,Ytr,Xte,Yte = du.loadData()

svm = SVMLoss.SVMLoss()

f = open("hyperdsearch.txt","w")

#iterate over cross validation folds...
for z in range(0,5):
    Xtr, Ytr, Xval, Yval = du.getTrainVal(Xtr,Ytr,z)
    
    svm.setData(Xtr,Ytr,Xval,Yval,Xte,Yte)

    W = svm.initScores(svm.Ytr,svm.Xtr_rows)

    print "transposing W - current shape is:"
    print W.shape
    W = W.transpose()
    print "W transposed shape:"
    print W.shape

    loss=0.

    #use 9 randomized values of each hyperpara,eter (there are two - step_size and reg strength)
    for a in range(0,9):
        step_size = 10 ** random.uniform(-3,3)
        for b in range(0,9):            
            reg = 10 ** random.uniform(-6,1)

            print "validation fold: %d" % z
            f.write("validation fold: %d\n" % z)

            print "Step size: %f" % step_size
            f.write("Step size: %f\n" % step_size)

            print "Reg strength: %f" % reg
            f.write("Reg strength: %f\n" % reg)



f.close()




'''
for a in range(0,9):
    for b in range(0,9):
        step-size = 10 ** random.uniform(-3,3)
        reg = 10 ** random.uniform(-6,1)
        f.write(a)
        f.write(b)
        f.write("Step size %f" % step_size)
        f.write("Reg %f" % reg)
        print a
        print b
        print "Step size %f" % step_size
        print "Reg %f" % reg

        for(epoch in range(0,4)):
'''

