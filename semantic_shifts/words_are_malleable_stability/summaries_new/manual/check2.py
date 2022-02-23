import os

with open('temp.txt', 'r', encoding='utf-8') as f:
    alllines = f.readlines()
    for line in alllines:
        if 'please' not in line:
            line = line[:-1]
            line = ''.join([c for c in line if not c.isdigit() and c != ':'])
            # print('\'\':\'{}\','.format(line[:-1]))
            print('\'{}\':\'\','.format(line[:-1]))