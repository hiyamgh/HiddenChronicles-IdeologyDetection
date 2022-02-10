from lang_trans.arabic import buckwalter
from nltk.corpus import wordnet as wn
import pyarabic.araby as araby
import pickle
# uncomment the below in order to run this script, if you dont have wordnet and omw installed
# import nltk
# nltk.download('wordnet')
# import nltk
# nltk.download('omw')

with open('ArSEL_ArSenL_database/ArSEL1.0.txt', 'r') as f:
    lines = f.readlines()

# AWN_OFFSET;EWN_OFFSET;POS_Tag;AWN_Lemma;SAMA_Lemma;Pos_Sentiment_score;Neg_Sentiment_Score;Confidence###AFRAID;AMUSED;ANGRY;ANNOYED;DONT_CARE;HAPPY;INSPIRED;SAD
emotions_sentiment_ar = {}
print('started reading emotions from ArSEL')
for line in lines[19:]:
    info = line.split(';')
    if info[3] != 'NA' and info[4] != 'NA':
        # this makes sure in case we have both SAMA and AWN present
        # we will just replace extra '-' in SAMA by ''
        # checked all cases lead to same transliteration
        print('????')
        print(buckwalter.untransliterate(info[3].split('_')[0]))
        print(buckwalter.untransliterate(info[4].split('_')[0]).replace('-', ''))
        print('????')

        # both untranslitrations yield same word at the end, so we'll rely on AWN's in case
        # both AWN and SAMA are found
        word = buckwalter.untransliterate(info[3].split('_')[0])
    else:
        if info[3] != 'NA':
            word = buckwalter.untransliterate(info[3].split('_')[0])
        else:
            word = buckwalter.untransliterate(info[4].split('_')[0]).replace('-', '')
        # araby.strip_diacritics(before_filter)

    # remove the diacritics from word
    word = araby.strip_diacritics(word)

    # get the sense and from part of speech and offset
    sense = wn.synset_from_pos_and_offset(info[2], int(info[1]))
    # sense = wn.synset_from_pos_and_offset(emotions[k]['POS_Tag'], emotions[k]['EWN_OFFSET'])
    # get the lemmas of that sense
    lemmas = [l.name() for l in sense.lemmas()]
    print("Lemmas for sense : " + sense.name() + "(" + sense.definition() + ") - " + str(lemmas))

    emotions_sentiment_ar[word] = {
        'AWN_OFFSET': info[0],
        'EWN_OFFSET': int(info[1]),
        'POS_Tag': info[2],
        'AWN_Lemma': info[3],
        'SAMA_Lemma': info[4],
        'Pos_Sentiment_score': float(info[5]),
        'Neg_Sentiment_Score': float(info[6]),
        'Confidence': int(info[7].split('###')[0]),
        'AFRAID': float(info[7].split('###')[1]),
        'AMUSED': float(info[8]),
        'ANGRY': float(info[9]),
        'ANNOYED': float(info[10]),
        'DONT_CARE': float(info[11]),
        'HAPPY': float(info[12]),
        'INSPIRED': float(info[13]),
        'SAD': float(info[14][:-1]),
        'sense': sense.name(),
        'sense_definition': sense.definition(),
        'lemmas': [l.name() for l in sense.lemmas()]
    }
print('finished reading emotions from ArSEL')
with open('ArSEL_ArSenL_database/emotions_sentiment_ar.pkl', 'wb') as handle:
    pickle.dump(emotions_sentiment_ar, handle, protocol=pickle.HIGHEST_PROTOCOL)

# for k in emotions:
#     print(k)
#     print(emotions[k]['AWN_Lemma'])
#     print(emotions[k]['SAMA_Lemma'])
#     print('AWN_Lemma: ', buckwalter.untransliterate(emotions[k]['AWN_Lemma'].split('_')[0]) if emotions[k]['AWN_Lemma'] != 'NA' else emotions[k]['AWN_Lemma'])
#     print('SAMA_Lemma: ', buckwalter.untransliterate(emotions[k]['SAMA_Lemma'].split('_')[0]) if emotions[k]['SAMA_Lemma'] != 'NA' else emotions[k]['SAMA_Lemma'])
    # # get the english wordnet offset
    # sense = wn.synset_from_pos_and_offset(emotions[k]['POS_Tag'], emotions[k]['EWN_OFFSET'])
    # lemmas = [l.name() for l in sense.lemmas()]
    # print("Lemmas for sense : " + sense.name() + "(" + sense.definition() + ") - " + str(lemmas))
    # # all_synsets = wn.synsets(offset_synset)
    # # # get lemmas
    # # for sense in all_synsets:
    # #     lemmas = [l.name() for l in sense.lemmas()]
    # #     print("Lemmas for sense : " + sense.name() + "(" + sense.definition() + ") - " + str(lemmas))
    # print('===========================================================')