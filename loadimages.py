from __future__ import division
from PIL import Image
import numpy as np
import csv
import numpy as np

#//load image from 'normalizedimages' dir and append into array
#//add index to array of indices
#//resulting in an array of n images (n=number of images in imagelabels.txt) x 2592 x 1935 
#//and a matching array with the image index in teh same position

#array to hold the image arrays...
#imgarray=np.empty(shape=(0,1935,2592,3))
#apparently quicker to append arrays to list then comvert that: (or find a way to vectorize...)
trainlist = []
testlist=[]
trainindarray=[]
testindarray=[]

i=0;

#let's put 15% into test set
#need at least 6 to start with, then need to keep track to ensure just 15% in are in test set


with open("./imglabels.txt") as imlabels:
    reader=csv.reader(imlabels,delimiter=",")
    curindex=0;
    numof=0;
    numintrain = 0
    numintest = 0
    for row in reader:
        addtotest=False        
	index=row[0]
	label = row[1]
	img = row[2]
	img_path = "./normalizedpics/" + img + ".JPG"
	im = Image.open(img_path)
	imar = np.asarray(im)	
	#add the imagarry to the array holding the images...
	#imgarray = np.append(imgarray,[imar],axis=0)
	if index==curindex:
            numof=numof+1
            print numof, numintest/numintrain
            if numof<6:
                numintrain=numintrain+1
            elif numintest/numintrain < 0.15:
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








  
