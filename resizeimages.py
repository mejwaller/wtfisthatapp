import numpy as np
import os
from PIL import Image

inpath="./dunpics/"
outpath="./normalizedpics/"

files=os.listdir(inpath)
#print files

for f in files:
    print f
    i=Image.open(inpath+f)
    if i.size != (2592,1935):
        print "resizing"
        i = i.resize((2592,1935))
    else:
        print "Not resizing"

    print "Saving " + f + " to " + outpath
    i.save(outpath+f)



