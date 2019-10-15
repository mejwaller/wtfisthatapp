import datutils as dat
import numpy as np

class SVMLoss(object):

#    def __INIT__(self):
#        self.du = dat.datutils()
    
    
    def loadData(self): 
        self.du = dat.datutils()

        self.Xtr,self.Ytr,self.Xte,self.Yte = self.du.loadData()

        self.Xtr = self.Xtr.astype('float64')
        self.Xte = self.Xte.astype('float64')

        self.Xtr,self.Xte = self.du.subMeanImg(self.Xtr,self.Xte)

        self.Xtr_rows, self.Xte_rows = self.du.flattenData(self.Xtr,self.Xte)

        self.Xtr_rows, self.Xte_rows = self.du.addBias(self.Xtr_rows,self.Xte_rows)

    def initScores(self):
        num_classes = len(set(self.Ytr))
        vec_length = self.Xtr_rows.shape[1]
        print num_classes
        print vec_length
        self.W = np.random.randn(num_classes,vec_length)*1e-5
        print self.W.shape



        



svm = SVMLoss()

svm.loadData()

svm.initScores()


