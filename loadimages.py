from PIL import Image
import numpy as np
import csv

with open("./imglabels.txt") as imlabels:
    reader=csv.reader(imlabels,delimiter=" ")
    for row in reader:
	img=row[0]
	index = row[1]
	label = row[2]
	img_path = "./normalizedpics/" + img + ".JPG"
	im = Image.open(img_path)

#//load image from 'normalizedimages' dir and append into array
#//add index to array of indices
#//resulting in an array of n images (n=number of images in imagelabels.txt) x 2592 x 1935 
#//and a matching array with the image index in teh same position
  
