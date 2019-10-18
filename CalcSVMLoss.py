import SVMLoss
import time

svm = SVMLoss.SVMLoss()

svm.loadData()

W = svm.initScores(svm.Ytr,svm.Xtr_rows)

loss=0.

tic=time.time()
for i in xrange(svm.Xtr_rows.shape[0]):
    print "Loss for example %d" % i
    loss+=svm.L_i(svm.Xtr_rows[i],svm.Ytr[i],W)

toc=time.time()
print "Total loss:"
print loss
print "and it took %fs to run" % (toc-tic)
