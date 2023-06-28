import sys

list_filename1 = sys.argv[1]
list_filename2 = sys.argv[2]

list1 = []
list2 = []

with open(list_filename1) as list_file1:
    list1 = list_file1.read().splitlines()

with open(list_filename2) as list_file2:
    list2 = list_file2.read().splitlines()

set2 = set(list2)
temp = [x for x in list1 if x not in set2]

for t in temp:
    print(t)
