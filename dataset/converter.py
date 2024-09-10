import csv


file = open('gowalla.txt', 'r')
target_file = open('gowalla.tsv', 'w')
lines = file.readlines()
for line in lines:
    splited = line.strip('\n').split(' ')
    src = splited[0]
    dst = splited[1:]
    for i in dst:
        target_file.write(src + '\t' + i + '\t' + '1' + '\n')