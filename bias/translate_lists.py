import os
from googletrans import Translator


def translate_words(txt_file):
    if not os.path.exists(txt_file):
        raise ValueError('File {} does not exist'.format(txt_file))

    translator = Translator()
    with open(txt_file, 'r') as f:
        # Using readlines()
        terms_english = f.readlines()
        with open('{}_arabic.txt'.format(txt_file[:-4]), 'w',  encoding="utf-8") as of:
            for term in terms_english:
                term_ar = translator.translate(term, dest='ar').text
                of.write(term_ar + '\n')


if __name__ == '__main__':
    # After translation, I have done manual verification
    # translate_words(txt_file='israeli_palestinian_conflict/israel_list_arabic.txt')
    # translate_words(txt_file='israeli_palestinian_conflict/non_occupation_practices_arabic.txt')
    # translate_words(txt_file='israeli_palestinian_conflict/occupation_practices_arabic.txt')

    # translate_words(txt_file='participants_aspect/participants_Israel.txt')
    translate_words(txt_file='participants_aspect/participants_palestine.txt')
    # translate_words(txt_file='military_aspect/methods_of_violence.txt')

    # translate_words(txt_file='terrorism/terrorism_list.txt')
    # translate_words(txt_file='occupation/occupation_list.txt')