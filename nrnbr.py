import datutils as dat
import numpy as np

class NearestNeighbour(object):

    def train(self,X,y):
        """
        N = number of (training) examples
        D = 'depth' which row size of image x column size of image x 3 (r,g,b)
        X is a N x D where each row is a (training) example. y is 1-dimension of size N (labels)
        """
        #the nearest neighbour classifier simply remembers all the training data
        self.Xtr=X
        self.ytr=y

    def predict(self, X):
        """
        N = number of (test) examples
        D = 'depth' which row size of image x column size of image x 3 (r,g,b)
        X is a N x D where each row is a (test) example.
        """

        num_test=X.shape[0]

        #make sure that the output type matches the input type
        Ypred =  np.zeros(num_test, dtype=self.ytr.dtype)

        #Loop over all test rows
        for i in xrange(num_test): #https://www.geeksforgeeks.org/range-vs-xrange-python/
            #find the nearest training image to the ith test image
            #using the L1 distance (sum of absolute value differences)
            print "Running test example %d" % (i)
            #distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
            distances = np.sum(np.square(self.Xtr - X[i,:]), axis=1)
            min_index = np.argmin(distances) #get the index with the smallest distance
            Ypred[i] = self.ytr[min_index] #predict the lable of the nearest example

        return Ypred

nn = NearestNeighbour()

du = dat.datutils()

Xtr, Ytr, Xte, Yte = du.loadData()

print Ytr
print len(set(Ytr))

print "Xtr shape: "
print Xtr.shape
print "Ytr shape: "
print Ytr.shape
print "Xte shape: "
print Xte.shape
print "Yte shape: "
print Yte.shape

print "subtracting mean img..."

#du.showMeanImg(Xtr)

Xtr=Xtr.astype('float64')
Xte=Xte.astype('float64')

Xtr, Xte = du.subMeanImg(Xtr,Xte)

Xtr = Xtr.astype('uint8')
Xte = Xte.astype('uint8')
i

print "Flattening data..."
#flatten out all images to be one-dimensional
Xtr_rows, Xte_rows = du.flattenData(Xtr,Xte)

print "Flattened Xtr shape:"
print Xtr_rows.shape
print "Flattened Xte shape:"
print Xte_rows.shape

nn.train(Xtr_rows, Ytr)
Yte_predict = nn.predict(Xte_rows)

print Yte_predict
print Yte

print 'accuracy: %f' % (np.mean(Yte_predict == Yte))
