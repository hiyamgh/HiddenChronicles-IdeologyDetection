import os

with open('nahar/summaries_azarbonyad.txt', 'r', encoding='utf-8') as f:
    sofar = set()
    alllines = f.readlines()
    for line in alllines:
        if 'please' not in line:
            line = ''.join([c for c in line if not c.isdigit() and c != ':' and c != '\n' and c != '\t'])
            if line.strip() != '' and line.strip() not in sofar and line.strip() not in ['saudiya', 'hariri', 'munazama', 'hezbollah', 'iran', 'falastini', 'syrian', 'arafat', 'mukawama']:
                print('\'{}\':\'\','.format(line.strip()))
                sofar.add(line.strip())
            # print('\'{}\':\'\','.format(line[:-1]))

    with open('assafir/summaries_azarbonyad.txt', 'r', encoding='utf-8') as f1:
        alllines = f1.readlines()
        for line in alllines:
            if 'please' not in line:
                line = ''.join([c for c in line if not c.isdigit() and c != ':' and c != '\n' and c != '\t'])
                if line.strip() != '' and line.strip() not in sofar and line.strip() not in ['saudiya', 'hariri',
                                                                                             'munazama', 'hezbollah',
                                                                                             'iran', 'falastini',
                                                                                             'syrian', 'arafat',
                                                                                             'mukawama']:
                    print('\'{}\':\'\','.format(line.strip()))
                    sofar.add(line.strip())
