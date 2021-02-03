from bias.embedding_biases import *
import getpass

if __name__ == '__main__':

    # get the keywords
    participant_israel = file_to_list(
        txt_file='israeli_palestinian_conflict/participants+methods_violence/participants_Israel_arabic.txt')
    participant_palestine = file_to_list(
        txt_file='israeli_palestinian_conflict/participants+methods_violence/participants_palestine_arabic.txt')
    methods_of_violence = file_to_list(
        txt_file='israeli_palestinian_conflict/participants+methods_violence/methods_of_violence_arabic.txt')

    archive = 'nahar'

    print('Getting Embedding Bias from {}'.format(archive))
    output_dir = 'outputs/participants_violence_eb/'
    if getpass.getuser() == '96171':
        get_embedding_bias(word_list1=participant_israel, word_list2=participant_palestine, neutral_list=methods_of_violence,
                           s_year=1933, e_year=1990, distype='cossim',
                           word2vec_models_path='E:/newspapers/word2vec/{}/embeddings/'.format(archive),
                           fig_name='{}_pisrael_ppalestine_violence'.format(archive),
                           output_folder=output_dir,
                           ylab='Israeli Violence')

        get_embedding_bias_decade_level(word_list1=participant_israel,
                                        word_list2=participant_palestine,
                                        neutral_list=methods_of_violence,
                                        decades_path='E:/newspapers/word2vec_decades/{}/meta_data/'.format(archive),
                                        archive=archive,
                                        fig_name='{}_pisrael_ppalestine_violence'.format(archive),
                                        output_folder=output_dir,
                                        ylab='Israeli Violence',
                                        distype='cossim')

    else:
        get_embedding_bias(word_list1=participant_israel, word_list2=participant_palestine, neutral_list=methods_of_violence,
                           s_year=1933, e_year=1990, distype='cossim',
                           word2vec_models_path='D:/word2vec/{}/embeddings/'.format(archive),
                           fig_name='{}_pisrael_ppalestine_violence'.format(archive),
                           output_folder=output_dir,
                           ylab='Israeli Violence')

        get_embedding_bias_decade_level(word_list1=participant_israel,
                                        word_list2=participant_palestine,
                                        neutral_list=methods_of_violence,
                                        decades_path='D:/word2vec_decades/{}/meta_data/'.format(archive),
                                        archive=archive,
                                        fig_name='{}_pisrael_ppalestine_violence_decade'.format(archive),
                                        output_folder=output_dir,
                                        ylab='Israeli Violence',
                                        distype='cossim')
