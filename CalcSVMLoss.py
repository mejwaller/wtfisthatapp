import SVMLoss
import time
import numpy as np

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
#tic = time.time()
step_size=2
reg=1e-3

delta=1e+9

scores = svm.score(W,svm.Xtr_rows)

orig_loss = 1e+9

while abs(delta) > 1:    

    loss, grad, scores = svm.SVM_loss(svm.Xtr_rows,svm.Ytr,W,reg)

    #toc=time.time()
    print "Total loss:"
    print loss
    #print "and it took %fs to run" % (toc-tic)

    predicted_class = np.argmax(scores, axis=1)    
    print "Trainign accurcay: %f" % (np.mean(predicted_class == svm.Ytr.astype('int64')))

    delta = orig_loss-loss

    W+=-step_size*grad

print "Predicted:"
print np.argmax(scores, axis=1)
print "Actual:"
print svm.Ytr.astype('int64')


'''
print "Calculating gradient..."
tic=time.time()
df = svm.eval_numerical_gradient(svm.unaryLoss,W)
toc = time.time()

print "Gradient is is: %f" % df
print "and it took %fs..." % (toc-tic)
'''
