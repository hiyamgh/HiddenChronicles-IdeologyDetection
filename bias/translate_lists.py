import os
from googletrans import Translator

'''
When the lists are translated, some translations become more than one word in arabic,
therefore each of these word becomes a separate word

NOTE: Unless we tell Word2Vec when we train it to take into account composite words 
'''


def translate_words(txt_file):
    file = os.path.join('keywords/english/', txt_file)
    if not os.path.exists(file):
        raise ValueError('File {} does not exist'.format(txt_file))

    translator = Translator()
    with open(file, 'r') as f:
        # get the terms in english
        terms_english = f.readlines()
        # create a file initially empty for putting the translations in arabic
        with open(os.path.join('keywords/arabic/{}'.format('{}_arabic.txt'.format(txt_file[:-4]))), 'w',  encoding="utf-8") as of:
            for term in terms_english:
                # translate the term
                term_ar = translator.translate(term, dest='ar').text
                # write the translated term on a new line in the file (each term is on a line)
                of.write(term_ar + '\n')


if __name__ == '__main__':
    # After translation, I have done manual verification
    # translate_words(txt_file='israeli_palestinian_conflict/israel_list_arabic.txt')
    # translate_words(txt_file='israeli_palestinian_conflict/non_occupation_practices_arabic.txt')
    # translate_words(txt_file='israeli_palestinian_conflict/occupation_practices_arabic.txt')

    # translate_words(txt_file='participants_aspect/participants_Israel.txt')
    # translate_words(txt_file='participants_aspect/participants_palestine.txt')
    # translate_words(txt_file='military_aspect/methods_of_violence.txt')

    # translate_words(txt_file='terrorism/terrorism_list.txt')
    # translate_words(txt_file='occupation/occupation_list.txt')

    files = ['break_from_violence.txt', 'diplomatic_routine.txt',
             'internal_affairs.txt', 'international_law.txt',
             'terrorism(100yearsofbias).txt']
    for f in files:
        translate_words(txt_file=f)