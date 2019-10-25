import datutils as dat
import numpy as np

class SVMLoss(object):

    def loadData(self): 
        self.du = dat.datutils()

        self.Xtr,self.Ytr,self.Xte,self.Yte = self.du.loadData()

        self.Xtr = self.Xtr.astype('float64')
        self.Xte = self.Xte.astype('float64')

        print "Calculating and subtracting mean img"
        self.Xtr,self.Xte = self.du.subMeanImg(self.Xtr,self.Xte)

        print "Flatteining data"
        self.Xtr_rows, self.Xte_rows = self.du.flattenData(self.Xtr,self.Xte)

        print "Adding bias"
        self.Xtr_rows, self.Xte_rows = self.du.addBias(self.Xtr_rows,self.Xte_rows)

    def initScores(self,Ytr,Xtr_rows):
        #print Ytr
        num_classes = len(set(Ytr))
        vec_length = Xtr_rows.shape[1]        
        #print num_classes
        #print vec_length
        W = np.random.randn(num_classes,vec_length)*1e-5
        #print W.shape
        return W
     
    def score(self,W,Xtr_rows):
        #print W.shape
        scores = Xtr_rows.dot(W)
        return scores

    def L_i(self,x, y, W):
        """
            unvectorized version. Compute the multiclass svm loss for a single example (x,y)
            - x is a column vector representing an image (e.g. 3073 x 1 in CIFAR-10)
            with an appended bias dimension in the 3073-rd position (i.e. bias trick)
            - y is an integer giving index of correct class (e.g. between 0 and 9 in CIFAR-10)
            - W is the weight matrix (e.g. 10 x 3073 in CIFAR-10)
        """
        delta = 1.0
        scores = self.score(W,x)
        correct_class_score = scores[y.astype('uint8')]
        D=W.shape[1]
        loss_i = 0

        for j in xrange(D):
            if j==y:#skip
                continue
            #accumulate loss for ith example
            loss_i+= max(0,scores[j] - correct_class_score + delta)
        return loss_i

    def SVM_loss(self,X,y,W,reg=1.,delta=1.):
        """
        fully-vectorized implementation :
        - X holds all the training examples as columns (e.g. 50,000  x 3073 in CIFAR-10)
        - y is array of integers specifying correct class (e.g. 50,000-D array)
        - W are weights (e.g. 3073 x 10)
        """
        loss=0.
        #print "X.shape[0] is %d" % X.shape[0]
        scores = self.score(W,X)
        correct_class_score = scores[np.arange(X.shape[0]),y.astype('uint8')]
        margins = np.maximum(0,scores - correct_class_score[:, np.newaxis]+delta)
        margins[np.arange(X.shape[0]),y.astype('uint8')]=0
        loss = np.sum(margins)
        #print "Raw loss is %f" % loss
        loss/=X.shape[0]
        #print "average loss is %f" % loss
        #print "regL2norm is %f" % self.regL2norm(W,reg)
        loss += self.regL2norm(W,reg)
        return loss

    def regL2norm(self,W,reg):
        #see http://cs231n.github.io/neural-networks-2/ for the reason for the 0.5...
        return 0.5*reg*np.sum(W*W)





