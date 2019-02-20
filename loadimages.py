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
imlist = []
indarray=[]

i=0;

with open("./imglabels.txt") as imlabels:
    reader=csv.reader(imlabels,delimiter=",")
    for row in reader:
        print row, i
	img=row[0]
	index = row[1]
	label = row[2]
	img_path = "./normalizedpics/" + img + ".JPG"
	im = Image.open(img_path)
	imar = np.asarray(im)	
	#add the imagarry to the array holding the images...
	#imgarray = np.append(imgarray,[imar],axis=0)
	imlist.append(imar)
	indarray.append(index)
	i+=1

imgarray=np.asarray(imlist)
#shape is (number of images) x 1935 (height) x 2592 (width) x 3 (RGB)
print imgarray.shape








  
