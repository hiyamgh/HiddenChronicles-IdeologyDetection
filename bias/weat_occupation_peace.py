from bias.embedding_biases import *
import getpass

if __name__ == '__main__':

    # get the keywords
    occupation_practices = file_to_list(txt_file='israeli_palestinian_conflict/occupations_vs_peace+israel/occupation_practices_arabic.txt')
    peace_practices = file_to_list(txt_file='israeli_palestinian_conflict/occupations_vs_peace+israel/non_occupation_practices_arabic.txt')
    israel_list = file_to_list(txt_file='israeli_palestinian_conflict/occupations_vs_peace+israel/israel_list_arabic.txt')
    palestine_list = file_to_list(txt_file='israeli_palestinian_conflict/occupations_vs_peace+israel/participants_palestine_arabic.txt')

    archive = 'nahar'

    print('Getting Embedding Bias from {}'.format(archive))
    output_dir = 'outputs_weat/peace_occupation/'
    if getpass.getuser() == '96171':
        # calculate_weat_bias(target_list1=israel_list,
        #                     target_list2=palestine_list,
        #                     attr_list1=occupation_practices,
        #                     attr_list2=peace_practices,
        #                     s_year=1933, e_year=1990,
        #                     word2vec_models_path='E:/newspapers/word2vec/{}/embeddings/'.format(archive))

        calculate_weat_bias_decade_level(target_list1=israel_list,
                            target_list2=palestine_list,
                            attr_list1=occupation_practices,
                            attr_list2=peace_practices,
                            decades_path='E:/newspapers/word2vec_decades/{}/meta_data/'.format(archive))


    else:
        # calculate_weat_bias(target_list1=israel_list,
        #                     target_list2=palestine_list,
        #                     attr_list1=occupation_practices,
        #                     attr_list2=peace_practices,
        #                     s_year=1933, e_year=1990,
        #                     word2vec_models_path='D:/word2vec/{}/embeddings/'.format(archive))

        calculate_weat_bias_decade_level(target_list1=israel_list,
                                         target_list2=palestine_list,
                                         attr_list1=occupation_practices,
                                         attr_list2=peace_practices,
                                         decades_path='D:/word2vec_decades/{}/meta_data/'.format(archive))

