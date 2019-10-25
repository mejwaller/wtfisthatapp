import unittest, SVMLoss
import numpy as np

class TestSVM(unittest.TestCase):

    def setUp(self):
        self.svm = SVMLoss.SVMLoss()
        self.Xtr_rows = np.array([[13,-7,11,1],[1,2,3,1]])
        #self.Ytr = np.array([0,1,1,1,2,2,2])
        self.Ytr = np.array([0,1])


    def testinitScores(self):
        W = self.svm.initScores(self.Ytr,self.Xtr_rows)
        expectedShape=(2,4)
        #print "testinitScores:"
        #print W.shape
        self.failUnless(W.shape == expectedShape)

    def testScore(self):
        W = np.eye(3,4).transpose()
        expected = np.array([[13,-7,11],[1,2,3]])
        """
        print "W:"
        print W
        print "expected"
        print expected
        print "expected.transpose():"
        print expected.transpose().astype('float64')
        print "score:"
        print self.svm.score(W,self.Xtr_rows)
        print "Done"
        """
        self.failUnless(np.array_equal(expected.astype('float64'),self.svm.score(W,self.Xtr_rows)))

    
    def testL_i(self):
        x = np.array([13,-7,11,1])
        W = np.eye(3,4).transpose()
        y=np.array(0)
        expected = 0. #max(0,-7-13+1) + max(0,11-13+1) where the 1 is delta
        self.failUnless(expected==self.svm.L_i(x,y,W))
        x = np.array([1,2,3,1])
        y=np.array(1)
        expected = 2. #max(0,1-2+1) + max(0,3-2+1)
        self.failUnless(expected==self.svm.L_i(x,y,W))
    
    def testL_itot(self):
        loss=0.
        expected = 251.81625
        
        W = np.asarray([[.2,1.5,0.],[-.5,1.3,.25],[.1,2.1,.2],[2,0.,-.3],[1.1,3.2,-1.2]])

        X = np.asarray([[56,231,24,2,1],[1,2,3,4,1],[4,3,2,1,1]])
        
        Y = np.asarray([0,1,2])

        for i in xrange(X.shape[0]):
            loss+=self.svm.L_i(np.asarray(X[i]),Y[i],W)
        loss/=X.shape[0]

        loss+=self.svm.regL2norm(W,1)

        '''
        W = np.eye(2,4).transpose()
        for i in xrange(self.Xtr_rows.shape[0]):
            loss+=self.svm.L_i(self.Xtr_rows[i],self.Ytr[i],W)
        loss/=self.Xtr_rows.shape[0]
        loss+=self.svm.regL2norm(W,1)
        '''
        print "TESTLI_TOT acutls score is: %f" % loss

        self.failUnless(np.isclose(expected,loss))

    def test_vectorized(self):
        loss=0.

        loss=0.
        expected = 251.81625
        
        W = np.asarray([[.2,1.5,0.],[-.5,1.3,.25],[.1,2.1,.2],[2,0.,-.3],[1.1,3.2,-1.2]])

        X = np.asarray([[56,231,24,2,1],[1,2,3,4,1],[4,3,2,1,1]])
        
        Y = np.asarray([0,1,2])
        
        loss = self.svm.SVM_loss(X,Y,W)
        print "testvectorized - actual loss is: %f" % loss
        #print loss
        self.failUnless(np.isclose(expected,loss))


if __name__ == '__main__': unittest.main()
