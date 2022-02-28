import pickle

years2words = {}
with open('../semantic_shifts/words_are_malleable_stability/summaries_new/manual/nahar/summaries_azarbonyad.txt', 'r', encoding='utf-8') as f:
    all_lines = f.readlines()
    for l in all_lines:
        if 'please' in l:
            year = l.split(':')[2][:-1] # get the year
            if year == '1985':
                print()
            if year not in years2words:
                years2words[year] = {}
            wordinside = l.split(':')[1]
            idx = wordinside.index('in year')
            word = wordinside[:idx] # get the word
            if word not in years2words[year]:
                years2words[year][word] = []
            else:
                word = word + '_1'
                years2words[year][word] = []
        else:
            if l == '\n':
                continue
            if '~~~~~~' in l:
                continue
            w_curr = l.strip()
            w_curr = w_curr[:-1]
            years2words[year][word].append(w_curr)

with open('year2word2summary_nahar.pickle', 'wb') as handle:
    pickle.dump(years2words, handle, protocol=pickle.HIGHEST_PROTOCOL)

years2words = {}
with open('../semantic_shifts/words_are_malleable_stability/summaries_new/manual/assafir/summaries_azarbonyad.txt', 'r', encoding='utf-8') as f:
    all_lines = f.readlines()
    for l in all_lines:
        if 'please' in l:
            year = l.split(':')[2][:-1] # get the year
            if year == '1985':
                print()
            if year not in years2words:
                years2words[year] = {}
            wordinside = l.split(':')[1]
            idx = wordinside.index('in year')
            word = wordinside[:idx] # get the word
            if word not in years2words[year]:
                years2words[year][word] = []
            else:
                word = word + '_1'
                years2words[year][word] = []
        else:
            if l == '\n':
                continue
            if '~~~~~~' in l:
                continue
            w_curr = l.strip()
            w_curr = w_curr[:-1]
            years2words[year][word].append(w_curr)

with open('year2word2summary_assafir.pickle', 'wb') as handle:
    pickle.dump(years2words, handle, protocol=pickle.HIGHEST_PROTOCOL)
