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

'''
print "Loss unvectorized version:"
tic=time.time()
#print "X.shape[0] is %d" % svm.Xtr_rows.shape[0]
for i in xrange(svm.Xtr_rows.shape[0]):
    print "Loss for example %d" % i
    loss+=svm.L_i(svm.Xtr_rows[i],svm.Ytr[i],W)

#print "Raw loss is: %f" % loss
loss/=svm.Xtr_rows.shape[0]
#print "Mean loss is %f" % loss
#print "regL2norm is %f" % svm.regL2norm(W,1.)
loss+=svm.regL2norm(W,1.)

toc=time.time()
print "Total loss:"
print loss
print "and it took %fs to run" % (toc-tic)
'''

print "Loss vectorized version:"
loss=0.
tic = time.time()

loss = svm.SVM_loss(svm.Xtr_rows,svm.Ytr,W)

toc=time.time()
print "Total loss:"
print loss
print "and it took %fs to run" % (toc-tic)

'''
print "Calculating gradient..."
tic=time.time()
df = svm.eval_numerical_gradient(svm.unaryLoss,W)
toc = time.time()

print "Gradient is is: %f" % df
print "and it took %fs..." % (toc-tic)
'''
