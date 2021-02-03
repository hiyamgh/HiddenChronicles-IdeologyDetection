import pandas as pd
import getpass
import random
from bias.utilities import *


def get_embedding_bias_decade_level(labels, word_lists, decades_path, output_folder,
                                    distype='norm', topKneighbs=3):
    missing_df = pd.DataFrame(columns=['decade', 'word list', 'missing_before', 'missing_after'])
    decades = get_decades(decades_path=decades_path)
    decades = sorted(decades, key= lambda x: x[0])
    missing_dict = {}
    for decade in decades:
        s_year = decade[0]
        e_year = decade[1]

        missing_dict[decade] = {}

        word2vec_currmodel = load_model_decade_level(decades_path, s_year, e_year)

        for i, word_list in enumerate(word_lists):
            missing_dict[decade][labels[i]] = {}

            # get the list of terms of those that are present in the vocabulary
            word_list_filtered, not_found = check_terms(word_list, word2vec_currmodel)

            # get percentage of missing words not found in the vocabulary from the original list
            percentage_missing_before = get_terms_missing_difference_percentage(original_terms=word_list,
                                            filtered_terms=word_list_filtered)

            # missing_dict[decade][labels[i]] = '{:.2f}%%'.format(percentage_missing)
            missing_dict[decade][labels[i]]['before'] = '{:.2f}%'.format(percentage_missing_before)

            # get words edit distances
            # if the word list of not found wrods is not empty
            if not_found:
                for t in not_found:
                    psb1 = edits1(t)
                    if any(psb1) in word2vec_currmodel.wv:
                        word_list_filtered.append(random.choice(psb1))
                    else:
                        psb2 = edits2(t)
                        if any(psb2) in word2vec_currmodel.wv:
                            word_list_filtered.append(random.choice(psb2))

            # get percentage of missing after adding words at 1/2 edit distanced away
            percentage_missing_after = get_terms_missing_difference_percentage(original_terms=word_list,
                                                                         filtered_terms=word_list_filtered)

            missing_dict[decade][labels[i]]['after'] = '{:.2f}%'.format(percentage_missing_after)

    for decade in missing_dict:
        for label in missing_dict[decade]:
            missing_df = missing_df.append({
                'decade': decade[0] + '-' + decade[1],
                'word list': label,
                'missing_before': missing_dict[decade][label]['before'],
                'missing_after': missing_dict[decade][label]['after']
            }, ignore_index=True)

    mkdir(output_folder)

    for word_list in labels:
        df = missing_df[missing_df['word list'] == word_list].drop(['word list'], axis=1)
        df = df.sort_values(by='decade')
        df.to_csv(os.path.join(output_folder, 'missing_{}_{}.csv'.format(word_list, archive)), index=False)


if __name__ == '__main__':

    israel_list = file_to_list(
        txt_file='israeli_palestinian_conflict/occupations_vs_peace+israel/israel_list_arabic.txt')
    palestine_list = file_to_list(
        txt_file='israeli_palestinian_conflict/occupations_vs_peace+israel/participants_palestine_arabic.txt')

    terrorism_list = file_to_list(txt_file='terrorism/terrorism_list_arabic.txt')

    methods_of_violence = file_to_list(
        txt_file='israeli_palestinian_conflict/participants+methods_violence/methods_of_violence_arabic.txt')

    occupation_practices = file_to_list(
        txt_file='israeli_palestinian_conflict/occupations_vs_peace+israel/occupation_practices_arabic.txt')
    peace_practices = file_to_list(
        txt_file='israeli_palestinian_conflict/occupations_vs_peace+israel/non_occupation_practices_arabic.txt')

    word_lists = [israel_list, palestine_list, terrorism_list, methods_of_violence, occupation_practices, peace_practices]
    labels = ['israel', 'palestine', 'terrorism', 'methods_violence', 'occupation', 'peace']

    archive = 'nahar'
    if getpass.getuser() == '96171':
        get_embedding_bias_decade_level(labels, word_lists,
                                        decades_path='E:/newspapers/word2vec_decades/{}/meta_data/'.format(archive),
                                        output_folder='outputs/missing/decade/')
    else:
        get_embedding_bias_decade_level(labels, word_lists,
                                        decades_path='D:/word2vec_decades/{}/meta_data/'.format(archive),
                                        output_folder='outputs/missing/decade/')