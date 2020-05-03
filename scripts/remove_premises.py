import sys

file = sys.argv[1]

for l in open(file, 'r'):
    pid, premise = l.strip().split("\t")
    print(f"{pid}\t.")

