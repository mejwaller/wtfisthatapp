import numpy as np
import os
import multiprocessing as mp
from PIL import Image

inpath="./dunpics/"
outpath="./normalizedpics/"

def resize(file):
    print file
    i=Image.open(inpath+file)
    if i.size != (2592,1935):
        print "resizing"
        i = i.resize((2592,1935))
    else:
        print "Not resizing"

    print "Saving " + file + " to " + outpath
    i.save(outpath+file)

if __name__ == '__main__':

    files=os.listdir(inpath)

    pool = mp.Pool(processes=8)

    results = [pool.apply_async(resize, args=(f,)) for f in files]
    output = [p.get() for p in results]

    #print(output) 




