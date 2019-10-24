import SVMLoss
import time

svm = SVMLoss.SVMLoss()

svm.loadData()

W = svm.initScores(svm.Ytr,svm.Xtr_rows)

print "transposing W - current shape is:"
print W.shape
W = W.transpose()
print "W transposed shape:"
print W.shape

loss=0.

tic=time.time()
for i in xrange(svm.Xtr_rows.shape[0]):
    print "Loss for example %d" % i
    loss+=svm.L_i(svm.Xtr_rows[i],svm.Ytr[i],W)

loss/=svm.Xtr_rows.shape[0]
loss+=svm.regL2norm(W,1.)

toc=time.time()
print "Total loss:"
print loss
print "and it took %fs to run" % (toc-tic)
