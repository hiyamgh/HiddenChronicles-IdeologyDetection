files = ['azarbonyad', 'gonen']
for file in files:
    with open('{}.txt'.format(file), 'r', encoding='utf-8') as fin:
        summaries = fin.readlines()
        summaries = [s[:-1] if '\n' in s else s for s in summaries]
        idxs = [i for i, e in enumerate(summaries) if e.strip() == ""]

        with open('{}_prp.txt'.format(file), 'w', encoding='utf-8') as f:
            f.write('{}={{\n'.format(file))
            all_keywords = {}
            for i in range(len(idxs)):
                if i == 0:
                    curr_list = summaries[:idxs[i]]
                else:
                    curr_list = summaries[idxs[i-1]+1:idxs[i]]

                keyword = curr_list[0].split(' ')[0]
                year = curr_list[0].split(' ')[1]
                if keyword not in all_keywords:
                    if all_keywords != {}:
                        f.write('\t}\n')
                    all_keywords[keyword] = []
                    f.write('\t\'{}\': {{\n'.format(keyword))

                f.write('\t\t\'{}\': {{\n'.format(year))
                for w in curr_list[1:]:
                    f.write('\t\t\t\'\' :\'{}\'\n'.format(w[:-1]))
                f.write('\t\t}\n')
                all_keywords[keyword].append(year)

            f.write('\t}\n')
            f.write('}\n')