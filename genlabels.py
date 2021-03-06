import csv
from collections import OrderedDict

labdict=dict()
imdict=dict()
countdict=dict()
i=0

with open("./classes.txt") as csvfile:
    reader=csv.reader(csvfile,delimiter=',',quotechar='"')
    for row in reader:
        print row
        if len(row) > 0:
            #print row
            label=row[1]
            imref=row[0]
            if label in labdict:
                print "Found label " + label
                imdict[imref]=labdict[label]
                print "Marking image " + imref + " as index " + str(labdict[label])
            else:
                print "Inserting " + label + " at index " + str(i)
                labdict[label]=i
                imdict[imref]=i
                print "Marking image " + imref + " as NEW index " + str(i)
                i+=1
csvfile.close()
outfile = open("imglabels.txt","w")

ordered = OrderedDict(sorted(imdict.items(), key=lambda t: t[1]))

val=0
count=0

for value in ordered.values():
    if value==val:
        count+=1
        #print value, count
    else:
        val+=1
        count=1
        #print value,count

    countdict[value]=count

print countdict
    

#print ordered.values()
#print len(ordered.values())
curchanged=False
cur=-1
i=-1
for things in ordered:
    if ordered[things] != cur:
        curchanged=True
    else:
        curchanged=False
    if countdict[ordered[things]] > 5:#only want images with 6 or more examples to ensure we have enough for test set
        #print countdict[ordered[things]]
        cur = ordered[things]
        #outfile.write(str(ordered[things]) + "," + labdict.keys()[labdict.values().index(ordered[things])] + "," + things +"\n")
        if curchanged==True:
            i+=1
        outfile.write(str(i) + "," + labdict.keys()[labdict.values().index(ordered[things])] + "," + things +"\n")

        
#for things in imdict:
    #outfile.write(things + "," + str(imdict[things]) + "," + labdict.keys()[labdict.values().index(imdict[things])] + "\n")
outfile.close()




