import pickle
from deep_translator import GoogleTranslator

# create all_summaries dictionary but for English
all_summaries_en = {}

with open('all_summaries.pkl', 'rb') as handle:
    all_summaries = pickle.load(handle)
for w in all_summaries:
    print(w)
    all_summaries_en[w] = {}
    for year in all_summaries[w]:
        all_summaries_en[w][year] = []
        print('YEAR: {}, w: {} ======================================================================================================'.format(year, w))
        for neigh in all_summaries[w][year]:
            if all_summaries[w][year][neigh] == []:
                print('skipping {} since it was not known in Arabic, so can\'t translate to english'.format(w))
            else:
                print(neigh)
                print(all_summaries[w][year][neigh])
                trans = GoogleTranslator(source='ar', target='en').translate(all_summaries[w][year][neigh][0])
                all_summaries_en[w][year].append(trans)
                print(trans)
                print('---------------------------------------- ')

        with open('all_summaries_en.pickle', 'wb') as handle:
            pickle.dump(all_summaries_en, handle, protocol=pickle.HIGHEST_PROTOCOL)

# load the english dictionary
with open('all_summaries_en.pkl', 'rb') as handle:
    all_summaries_en = pickle.load(handle)

