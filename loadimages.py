from PIL import Image
import numpy as np
import csv

with open("./imagelabels,txt") as imlabels:
    reader=csv.reader(imlabels,delimiter=" ")
    for row in reader:
//load image from 'normalizedimages' dir and append into array
//add index to array of indices
//resulting in an array of n images (n=number of images in imagelabels.txt) x 2592 x 1935 
//and a matching array with the image index in teh same position
  
