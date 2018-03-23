#notes - mixing dicst and lists here - wip

import csv

labdict=dict()
imdict=dict()
i=0

with open("./classes.txt") as csvfile:
    reader=csv.reader(csvfile,delimiter=',',quotechar='"')
    for row in reader:
        if len(row) > 0:
            #print row
            label=row[2]
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
for things in imdict:
    outfile.write(things + " " + str(imdict[things]) + "\n")
outfile.close()




