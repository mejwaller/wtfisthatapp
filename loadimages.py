from __future__ import division
from PIL import Image
import numpy as np
import csv
import numpy as np
import pickle

#//load image from 'normalizedimages' dir and append into array
#//add index to array of indices
#//resulting in an array of n images (n=number of images in imagelabels.txt) x 2592 x 1935 
#//and a matching array with the image index in teh same position

#array to hold the image arrays...
#imgarray=np.empty(shape=(0,1935,2592,3))
#apparently quicker to append arrays to list then comvert that: (or find a way to vectorize...)
trainlist = []
testlist=[]
trainindlist=[]
testindlist=[]

i=0;

#let's put 15% into test set
#need at least 6 to start with, then need to keep track to ensure just 15% in are in test set

#two changes ot make:
# 1: Only add images ot training data if there are 6 or more of them
# 2: Only add 50 at a time and then pickle to save havoing am assive set in memory in once.
#Will have seperate proigram to load pickle files and append into one for when we do clasification bit.



with open("./imglabels.txt") as imlabels:
    reader=csv.reader(imlabels,delimiter=",")
    curindex=0;
    numof=0;
    numfortrain = 0
    numintest = 0
    sizeoftrain=0
    pklnum=1
    for row in reader:
        addtotest=False        
        addtotrain=False
	index=row[0]
	label = row[1]
	img = row[2]
	img_path = "./normalizedpics/" + img + ".JPG"

        if index==curindex:#imglabels.txt is ordedered by index so this condition is true only if next row is same species
            numof=numof+1
            if numof<6: #enough to put 15% in trainign set
                addtotrain=True
                numfortrain=numfortrain+1
            else:
                if numintest/numfortrain < 0.15:#if we already have more than 15% in test set, don't add more, keep them for training!
                    addtotrain=False;
                    addtotest=True;
                    numintest+=1
                else:
                    addtotrain=True;
                    addtotest=False;
                    numfortrain=numfortrain+1

        else:#new species
            curindex=index
            numof=1
            numfortrain=1
            numintest=0
            addtotest=False
            addtotrain=True

        if addtotrain:
            im = Image.open(img_path)
            imar = np.asarray(im)
            trainlist.append(imar)
            trainindlist.append(index)
            sizeoftrain+=1
        elif addtotest:#remember, test array will necve get massive, so no need to split
            im = Image.open(img_path)
            imar = np.asarray(im)
            testlist.append(imar)
            testindlist.append(index)

        print row, i, addtotrain, addtotest, sizeoftrain
            
        if(sizeoftrain==50):#pickle it, and then reset trainlist and trainindarray
            #and what if we've reached the end but sizeoftrain != 50?
            x = 'train' + str(pklnum) + '.pkl'
            y = 'indexes' + str(pklnum) + '.pkl'
            train=open(x,'wb')
            indexes=open(y,'wb')
            trainarray=np.asarray(trainlist)
            trainindarray=np.asarray(trainindlist)
            pickle.dump(trainarray,train)
            pickle.dump(trainindarray,indexes)
            train.close()
            indexes.close()
            print trainarray.shape
            print trainindarray.shape
            print "Pickled " + x + " and " + y
            trainlist=[]
            trainindlist=[]
            sizeoftrain=0
            pklnum+=1
        

    
        #print row, i, addtotrain, addtotest, sizeoftrain

        i+=1

    #now pickel any remaining test (i.e. we've finsihed reows but sizeoftrain < 50)
    if sizeoftrain > 0:
        x = 'train' + str(pklnum) + '.pkl'
        y = 'indexes' + str(pklnum) + '.pkl'
        train=open(x,'wb')
        indexes=open(y,'wb')
        trainarray=np.asarray(trainlist)
        trainindarray=np.asarray(trainindlist)
        pickle.dump(trainarray,train)
        pickle.dump(trainindarray,indexes)
        train.close()
        indexes.close()
        print trainarray.shape
        print trainindarray.shape
        print "Pickled " + x + " and " + y
        trainlist=[]
        trainindlist=[]
        sizeoftrain=0
        pklnum+=1


testarray = np.asarray(testlist)
testindarray = np.asarray(testindlist)

print testarray.shape
print testindarray.shape

test=open('test.pkl','wb')
testinds=open('testinds.pkl','wb')
pickle.dump(testarray,test)
pickle.dump(testindarray,testinds)
test.close()
testinds.close()

print "Pickled test.pkl and testinds.pkl"


            






'''
        im = Image.open(img_path)
	imar = np.asarray(im)	
	#add the imagarry to the array holding the images...
	#imgarray = np.append(imgarray,[imar],axis=0)
        #set aside 15% as a test set - need enoough in train that a 15% test set can be extracted
        #leaves question about how do we know how good it *really* is for where there's not enough i
        #of a given speciesi in train to get a test set out?
	if index==curindex:#imglabels.txt is ordedered by index so this condition is true only if next row is same species
            numof=numof+1
            print numof, numintest/numintrain
            if numof<6: #not enough for 15% (1/6 ~ 1.7)
                numintrain=numintrain+1
            elif numintest/numintrain < 0.15:#if we already have more than 15% in test set, don;t add more, keep them for training!
                addtotest=True
                numintest=numintest+1
            else:
                addtotest=False
                numintrain=numintrain+1
	else:
            #store 'numof' for previous index
            curindex=index
            numof=1
            numintrain=1
            numintest=0
            addtotest=False
            
	print row, i, addtotest
	
	if addtotest:
            testlist.append(imar)
            testindarray.append(index)
        else:
            trainlist.append(imar)
            trainindarray.append(index)
	i+=1


trainarray=np.asarray(trainlist)
testarray=np.asarray(testlist)
#shape is (number of images) x 1935 (height) x 2592 (width) x 3 (RGB)
print trainarray.shape
print testarray.shape
#todo - store arrays (pickle?)
test= open('test.pkl','wb')

pickle.dump(testarray,test)

test.close()

print 'pickled test data.'

remainder=0

for i in range(1,((int)(trainarray.shape[0]/50 + 1))):
    x = 'train' + str(i) + '.pkl'
    print x
    train=open(x,'wb')
    startidx = (i-1)*50
    endidx = startidx+49+1
    print startidx,endidx
    pickle.dump(trainarray[startidx:endidx,],train)
    train.close()
    
    remainder=i

    
startidx=remainder*50

x='train'+str(remainder+1)+'.pkl'
print x

train=open(x,'wb')

print startidx,endidx

pickle.dump(trainarray[startidx:,],train)

train.close()

print 'pickled training data.'

'''









  
