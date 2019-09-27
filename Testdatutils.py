import unittest
import datutils as dat
import numpy as np


class myTest(unittest.TestCase):

    def setUp(self):
        #setup some simpel 3x3 pcitures
        self.pic1 = np.array([[1,2,3],[3,4,5],[6,7,8]])
        self.pic2 = np.array([[1,2,3],[1,2,3],[1,2,3]])
        self.pic3 = np.array([[3,4,5],[3,4,5],[3,4,5]])
        self.pic4 = np.array([[7,8,9],[7,8,9],[7,8,9]])

        self.traindata=[]
        self.traindata.append(self.pic1)
        self.traindata.append(self.pic2)
        self.traindata.append(self.pic3)
        self.traindata.append(self.pic4)

        self.traindata = np.array(self.traindata)

        self.pic5 = np.array([[1,2,3],[4,5,6],[7,8,9]])
        self.pic6 = np.array([[2,3,4],[5,6,7],[8,9,0]])
        self.pic7 = np.array([[3,4,5],[6,7,8],[9,0,1]])

        self.testdata=[]
        self.testdata.append(self.pic5)
        self.testdata.append(self.pic6)
        self.testdata.append(self.pic7)

        self.testdata = np.array(self.testdata)

        self.du = dat.datutils()

    def testMeanImg(self):
        expected = np.array([[3.,4.,5.],[3.5,4.5,5.5],[4.25,5.25,6.25]])
        self.failUnless(np.array_equal(expected,self.du.getMeanImg(self.traindata)))

    def testsubMeanImg(self):
        Xexpected = [[[-2.,-2.,-2.],[-0.5,-0.5,-0.5],[1.75,1.75,1.75]],[[-2.,-2.,-2.],[-2.5,-2.5,-2.5],[-3.25,-3.25,-3.25]],[[0.,0.,0.],[-0.5,-0.5,-0.5],[-1.25,-1.25,-1.25]],[[4.,4.,4.],[3.5,3.5,3.5],[2.75,2.75,2.75]]]
        Yexpected = [[[-2.,-2.,-2.],[0.5,0.5,0.5],[2.75,2.75,2.75]],[[-1.,-1.,-1.],[1.5,1.5,1.5],[3.75,3.75,-6.25]],[[0.,0.,0.],[2.5,2.5,2.5],[4.75,-5.25,-5.25]]]

        meanImg = np.asarray(self.du.getMeanImg(self.traindata))
        Xsubmean = self.traindata - meanImg
        Ysubmean = self.testdata - meanImg

        self.failUnless(np.array_equal(Xexpected,Xsubmean))
        self.failUnless(np.array_equal(Yexpected,Ysubmean))

'''
    def testaddBias(self):
        meanImg=np.asarray(self.du.getMeanImg(self.traindata))
        Xsubmean=self.traindata - meanImg
        Ysubmean=self.testdata - meanImg

        print Xsubmean.shape
        print Ysubmean.shape

        print "np.omnes..."

        print np.ones((Xsubmean.shape[2],1))
        print np.ones((Ysubmean.shape[2],1))

        Xtrain = np.hstack([Xsubmean,np.ones((Xsubmean.shape[2],1))])
        Ytrain = np.hastack([Ysubmean,np.ones((Ysubmean.shape[2],1))])

        print Xtrain
        print Ytrain
'''
        





if __name__ == '__main__': unittest.main()   

        
