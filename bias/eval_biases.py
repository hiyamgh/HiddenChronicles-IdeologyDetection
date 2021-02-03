from bias.embedding_biases import *
import getpass

if __name__ == '__main__':
    # translate_words('occupations1950_english.txt')
    # translate_words('male_pairs.txt')
    # translate_words('female_pairs.txt')
    # add_gender_nouns(txt_file='occupations1950_arabic_mod.txt')

    # word2vec_models = load_models(start_year=1950, end_year=1976, archive_path='D:/word2vec/hayat/embeddings/')
    archive = 'nahar'
    male_wordlist = file_to_list(txt_file='male_pairs_arabic.txt')
    female_wordlist = file_to_list(txt_file='female_pairs_arabic.txt')
    occupations_wordlist = file_to_list(txt_file='occupations1950_arabic_mod_gender_based.txt')

    occupation_practices = file_to_list(txt_file='israeli_palestinian_conflict/occupations_vs_peace+israel/occupation_practices_arabic.txt')
    peace_practices = file_to_list(txt_file='israeli_palestinian_conflict/occupations_vs_peace+israel/non_occupation_practices_arabic.txt')
    israel_list = file_to_list(txt_file='israeli_palestinian_conflict/occupations_vs_peace+israel/israel_list_arabic.txt')

    # participants
    participant_israel = file_to_list(txt_file='israeli_palestinian_conflict/participants+methods_violence/participants_Israel_arabic.txt')
    participant_palestine = file_to_list(txt_file='israeli_palestinian_conflict/participants+methods_violence/participants_palestine_arabic.txt')

    # methods of violence
    methods_of_violence = file_to_list(txt_file='israeli_palestinian_conflict/participants+methods_violence/methods_of_violence_arabic.txt')

    # occupation words
    occupation_list = file_to_list(txt_file='terrorism/terrorism_list_arabic.txt')

    # terrorism words
    terrorism_list = file_to_list(txt_file='terrorism/terrorism_list.txt')

    # get embedding Bias Women vs. Men Occupation - Annahar Archive
    if getpass.getuser() == '96171':
        # print('Getting Embedding Bias from {}'.format(archive))
        # get_embedding_bias(word_list1=female_wordlist, word_list2=male_wordlist, neutral_list=occupations_wordlist,
        #                    s_year=1933, e_year=1990, distype='cossim',
        #                    word2vec_models_path='E:/newspapers/word2vec/{}/embeddings/'.format(archive), fig_name='{}_eb'.format(archive),
        #                    ylab='Women')

        print('Getting Embedding Bias from {}'.format(archive))
        get_embedding_bias(word_list1=peace_practices, word_list2=occupation_practices, neutral_list=israel_list,
                           s_year=1933, e_year=1990, distype='cossim',
                           word2vec_models_path='E:/newspapers/word2vec/{}/embeddings/'.format(archive),
                           fig_name='{}_israel_palestine'.format(archive),
                           ylab='Peace Practices')

        print('Getting Embedding Bias from {}'.format(archive))
        get_embedding_bias(word_list1=participant_israel, word_list2=participant_palestine,
                           neutral_list=methods_of_violence,
                           s_year=1933, e_year=1990, distype='cossim',
                           word2vec_models_path='E:/newspapers/word2vec/{}/embeddings/'.format(archive),
                           fig_name='participants_methods_of_violence_{}'.format(archive),
                           ylab='Israeli Violence')

    else:
        # print('Getting Embedding Bias from Annahar')
        # get_embedding_bias(word_list1=female_wordlist, word_list2=male_wordlist, neutral_list=occupations_wordlist,
        #                    s_year=1933, e_year=1990, distype='cossim',
        #                    word2vec_models_path='D:/word2vec/nahar/embeddings/', fig_name='nahar_eb')

        print('Getting Embedding Bias from {}'.format(archive))
        get_embedding_bias(word_list1=peace_practices, word_list2=occupation_practices, neutral_list=israel_list,
                           s_year=1933, e_year=1990, distype='cossim',
                           word2vec_models_path='D:/word2vec/{}/embeddings/'.format(archive),
                           fig_name='{}_israel_palestine_hiyam'.format(archive),
                           ylab='Peace Practices')

        # print('Getting Embedding Bias from {}'.format(archive))
        # get_embedding_bias_decade_level(word_list1=peace_practices,
        #                                 word_list2=occupation_practices,
        #                                 neutral_list=israel_list,
        #                                 decades_path='D:/word2vec_decades/{}/meta_data/'.format(archive),
        #                                 archive=archive,
        #                                 fig_name='{}_israel_palestine_decade'.format(archive),
        #                                 ylab='Peace Practices',
        #                                 distype='cossim')

        # print('Getting Embedding Bias from {}'.format(archive))
        # get_embedding_bias(word_list1=participant_israel, word_list2=participant_palestine,
        #                    neutral_list=methods_of_violence,
        #                    s_year=1933, e_year=1990, distype='cossim',
        #                    word2vec_models_path='D:/word2vec/{}/embeddings/'.format(archive),
        #                    fig_name='participants_methods_of_violence_{}'.format(archive),
        #                    ylab='Israeli Violence')

        # print('Getting Embedding Bias from {}'.format(archive))
        # get_embedding_bias(word_list1=participant_israel, word_list2=participant_palestine,
        #                    neutral_list=occupation_list,
        #                    s_year=1933, e_year=1990, distype='cossim',
        #                    word2vec_models_path='D:/word2vec/{}/embeddings/'.format(archive),
        #                    fig_name='participants_occupation_{}'.format(archive),
        #                    ylab='Israeli Occupation')

        # # THIS ONE
        # print('Getting Embedding Bias from {}'.format(archive))
        # get_embedding_bias_decade_level(word_list1=participant_israel,
        #                                 word_list2=participant_palestine,
        #                                 neutral_list=occupation_list,
        #                                 decades_path='D:/word2vec_decades/{}/meta_data/'.format(archive),
        #                                 archive=archive,
        #                                 fig_name='participants_occupation_{}_decades'.format(archive),
        #                                 ylab='Israeli Occupation',
        #                                 distype='cossim')

        # print('Getting Embedding Bias from {}'.format(archive))
        # get_embedding_bias_decade_level(word_list1=participant_israel,
        #                                 word_list2=participant_palestine,
        #                                 neutral_list=terrorism_list,
        #                                 decades_path='D:/word2vec_decades/{}/meta_data/'.format(archive),
        #                                 archive=archive,
        #                                 fig_name='participants_terrorism_{}_decades'.format(archive),
        #                                 ylab='Israeli Terrorism',
        #                                 distype='cossim')


        # print('Getting Embedding Bias from {}'.format(archive))
        # get_embedding_bias(word_list1=participant_israel, word_list2=participant_palestine,
        #                    neutral_list=terrorism_list,
        #                    s_year=1933, e_year=1990, distype='cossim',
        #                    word2vec_models_path='D:/word2vec/{}/embeddings/'.format(archive),
        #                    fig_name='participants_terrorism_{}'.format(archive),
        #                    ylab='Israeli Terrorism')