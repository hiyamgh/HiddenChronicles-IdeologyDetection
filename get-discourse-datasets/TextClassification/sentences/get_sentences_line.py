import os

with open('sentences.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

    with open('TextLine.txt', 'w', encoding='utf-8') as fout:
        for line in lines:
            if line.strip() == '':
                continue
            if 'group' in line.strip():
                continue
            if line.strip().isdigit():
                continue

            fout.write(line)