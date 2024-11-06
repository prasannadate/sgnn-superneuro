import os
import sys

f = open("Pubmed-Diabetes.DIRECTED.cites.tab", 'r')
lines = f.readlines()
f.close()

lines = lines[2:]
for line in lines:
    fields = line.split()
    print(fields[1] + "," + fields[3])


