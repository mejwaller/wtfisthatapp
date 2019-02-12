from PIL import Image
import numpy as np
import csv
import numpy as np

#//load image from 'normalizedimages' dir and append into array
#//add index to array of indices
#//resulting in an array of n images (n=number of images in imagelabels.txt) x 2592 x 1935 
#//and a matching array with the image index in teh same position

imgarray=[]
indarray=[]

i=0;

with open("./imglabels.txt") as imlabels:
    reader=csv.reader(imlabels,delimiter=",")
    for row in reader:
        #print row
	img=row[0]
	index = row[1]
	label = row[2]
	img_path = "./normalizedpics/" + img + ".JPG"
	im = Image.open(img_path)
	imgarray.append(im)
	indarray.append(index)
	i+=1

iar = np.asarray(imgarray[1])

print iar

#shape is 1935 (height) x 2592 (width) x 3 (RGB)
print iar.shape
print imgarray[1]





  
