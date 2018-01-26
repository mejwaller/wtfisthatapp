#notes - mixing dicst and lists here - wip

import csv

labdict=dict()

with open("/home/mejwaller/classes.txt") as csvfile:
    reader=csv.reader(csvfile,delimiter=',',quotechar='"')
    for row in reader:
        if len(row) > 0:
            #print row
            number = row[1]
            label=row[2]
            if label in labdict.values():
                print "Found label " + label
                i=lablist.
                if i != int(number):
                    print "error - " + label + " exists at index " + str(i) + " but in file it is given index " + number
                    raise SystemExit
            else:
                lablist.insert(int(number)-1,label)



