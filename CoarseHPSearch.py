import datutils as dat
import random
import SVMLoss
import numpy as np

#hyperparameters: reg, step size
#use 5 epochs for corase search!

#9 examples of each, randomized

du = dat.datutils()

Xtr_base,Ytr_base,Xte_base,Yte_base = du.loadData()

svm = SVMLoss.SVMLoss()

print "Generating validation folds"

du.genXvalFolds(Xtr_base,Ytr_base)

Xtr_base=[]
Ytr_base=[]

del Xtr_base
del Ytr_base


f = open("hypersearch.csv","w")

f.write("step_num,reg_num,validation fold,step_size,reg,loss,train_acc,val_acc\n")

#use 9 randomized values of each hyperpara,eter (there are two - step_size and reg strength)
for a in range(0,9):
    step_size = 10 ** random.uniform(-3,3)
    for b in range(0,9):            
        reg = 10 ** random.uniform(-6,1)

        #iterate over cross validation folds...
        for z in range(0,5):

            Xtr,Ytr,Xval,Yval = du.getTrainVal(z)
    
            svm.setData(Xtr,Ytr,Xval,Yval,Xte_base,Yte_base)

            W = svm.initScores(svm.Ytr,svm.Xtr_rows)

            print a
            print b

            #print "transposing W - current shape is:"
            #print W.shape
            W = W.transpose()
            #print "W transposed shape:"
            #print W.shape

            loss=0.

            print "validation fold: %d" % z            

            print "Step size: %f" % step_size

            print "Reg strength: %f" % reg

            fname = str(a)+"_"+str(b) + "_"+str(z)+".csv"

            f2 = open(fname,"w")

            f2.write("epoch,step_size,reg,loss,train_acc,val_acc\n")

            for epoch in range(0,9):

                print "Epoch %d" % epoch

                loss,grad,scores = svm.SVM_loss(svm.Xtr_rows,svm.Ytr,W,reg)
    
                predicted_class = np.argmax(scores,axis=1)
                train_acc = (np.mean(predicted_class == svm.Ytr.astype('int64')))                
    
                #print "Training accurcay after epoch %d is %f" % (epoch,train_acc)

                #print "and loss is %f" % loss

                W+=step_size*grad

                #validation accuracy

                valloss, valgrad, valscores = svm.SVM_loss(svm.Xval_rows,svm.Yval,W,reg)

                predicted_class = np.argmax(valscores,axis=1)
                val_acc = np.mean(predicted_class == svm.Yval.astype('int64'))

                #print " and validation accurcay is %f" % val_acc

                f2str = str(epoch) + "," + str(step_size)+"," + str(reg) + "," + str(loss) + "," + str(train_acc) + "," + str(val_acc) + "\n"

                f2.write(f2str)

            fstr = str(a) + "," + str(b) + "," + str(z) + "," + str(step_size) + "," + str(reg) + "," + str(loss) + "," + str(train_acc) + "," + str(val_acc) + "\n"

            f.write(fstr)

            f2.close()


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

