from bias.embedding_biases import *
import getpass
from bias.plotting import *

if getpass.getuser() == '96171':
    archives_wordembeddings = {
        'nahar': 'E:/newspapers/word2vec/nahar/embeddings/',
        'hayat': 'E:/newspapers/word2vec/hayat/embeddings/',
        'assafir': 'E:/newspapers/word2vec/assafir/embeddings/'
    }
else:
    archives_wordembeddings = {
        'nahar': 'D:/word2vec/nahar/embeddings/',
        'hayat': 'D:/word2vec/hayat/embeddings/',
        'assafir': 'D:/word2vec/assafir/embeddings/'
    }

# # get the keywords
#     participant_israel = file_to_list(
#         txt_file='israeli_palestinian_conflict/participants+methods_violence/participants_Israel_arabic.txt')
#
#     participant_palestine = file_to_list(
#         txt_file='israeli_palestinian_conflict/participants+methods_violence/participants_palestine_arabic.txt')
#
#     occupation_list = file_to_list(txt_file='occupation/occupation_list_arabic.txt')

# # get the keywords
#     participant_israel = file_to_list(
#         txt_file='israeli_palestinian_conflict/participants+methods_violence/participants_Israel_arabic.txt')
#
#     participant_palestine = file_to_list(
#         txt_file='israeli_palestinian_conflict/participants+methods_violence/participants_palestine_arabic.txt')
#
#     terrorism_list = file_to_list(txt_file='terrorism/terrorism_list_arabic.txt')

# # get the keywords
#     participant_israel = file_to_list(
#         txt_file='israeli_palestinian_conflict/participants+methods_violence/participants_Israel_arabic.txt')
#     participant_palestine = file_to_list(
#         txt_file='israeli_palestinian_conflict/participants+methods_violence/participants_palestine_arabic.txt')
#     methods_of_violence = file_to_list(
#         txt_file='israeli_palestinian_conflict/participants+methods_violence/methods_of_violence_arabic.txt')

if __name__ == '__main__':

    # peace vs. occupation - israel
    # participant israel vs. participant palestine - occupation
    # participant israel vs. participant palestine - terrorism
    # participant israel vs. participant palestine - methods of violence

    # get the keywords
    main_dir = 'israeli_palestinian_conflict/occupations_vs_peace+israel/'
    peace_practices = file_to_list(txt_file=os.path.join(main_dir, 'non_occupation_practices_arabic.txt'))
    occupation_practices = file_to_list(txt_file=os.path.join(main_dir, 'occupation_practices_arabic.txt'))
    israel_list = file_to_list(txt_file=os.path.join(main_dir, 'israel_list_arabic.txt'))
    palestine_list = file_to_list(txt_file=os.path.join(main_dir, 'participants_palestine_arabic.txt'))

    # set 2
    # participants israel vs. participants palestine
    main_dir2 = 'israeli_palestinian_conflict/participants+methods_violence/'
    participants_israel = file_to_list(txt_file=os.path.join(main_dir2, 'participants_Israel_arabic.txt'))
    participants_palestine = file_to_list(txt_file=os.path.join(main_dir2, 'participants_palestine_arabic.txt'))
    occupation_list = file_to_list(txt_file='occupation/occupation_list_arabic.txt')

    # set 3
    # participants israel vs. participants palestine
    terrorism_list = file_to_list(txt_file='terrorism/terrorism_list_arabic.txt')

    # set 4
    # participants israel vs. participants palestine
    methods_of_violence = file_to_list(txt_file=os.path.join(main_dir2, 'methods_of_violence_arabic.txt'))

    # output_dir = 'outputs/peace_occupation_eb/'
    #
    # target_list1 = [peace_practices, participants_israel, participants_israel, participants_israel]
    # target_list2 = [occupation_practices, participants_palestine, participants_palestine, participants_palestine]
    # neutral_lists = [israel_list, occupation_list, terrorism_list, methods_of_violence]
    # fig_names = ['peace_occupation_israel', 'israel_palestine_occupation', 'israel_palestine_terrorism',
    #              'israel_palestine_mov']
    # ylabs = ['Avg. Embedding Bias Peace Practices', 'Avg. Embedding Bias Israeli Occupation',
    #          'Avg. Embedding Bias Israeli Terrorism', 'Avg. Embedding Bias Israeli Methods of Violence']
    #
    # all_embedding_biases = get_embedding_bias(word_lists1=target_list1, word_lists2=target_list2,
    #                                           neutral_lists=neutral_lists,
    #                                           desired_archives=['nahar', 'hayat', 'assafir'],
    #                                           # desired_archives=['hayat'],
    #                                           #  desired_archives=['hayat'],
    #                                           wemb_path=archives_wordembeddings,
    #                                           distype='cossim')
    # plot_embedding_bias_time(all_embedding_biases, 'latest', fig_names, ylabs)

    target_list1 = [participants_israel, participants_israel, participants_israel]
    target_list2 = [participants_palestine, participants_palestine, participants_palestine]
    neutral_lists = [occupation_list, terrorism_list, methods_of_violence]
    # for archive in ['nahar', 'hayat']:
    # for archive in ['nahar']:
    #     get_changing_attitudes(word_lists1=target_list1, word_lists2=target_list2,
    #                            neutral_lists=neutral_lists, archive=archive, desired_years=(1960, 1970),
    #                            wemb_path=archives_wordembeddings,
    #                            output_dir='latest', fig_name='israel_palestine_attitude_{}'.format(archive),
    #                            distype='norm', topKneighbs=3)
    #
        # get_changing_attitudes(word_lists1=target_list1, word_lists2=target_list2,
        #                        neutral_lists=neutral_lists, archive=archive, desired_years=(1940, 1950),
        #                        wemb_path=archives_wordembeddings,
        #                        output_dir='latest', fig_name='israel_palestine_attitude_{}_{}-{}'.format(archive,
        #                                                                                                  1940, 1950),
        #                        distype='norm', topKneighbs=3)
    #
    #     get_changing_attitudes(word_lists1=target_list1, word_lists2=target_list2,
    #                            neutral_lists=neutral_lists, archive=archive, desired_years=(1974, 1984),
    #                            wemb_path=archives_wordembeddings,
    #                            output_dir='latest', fig_name='israel_palestine_attitude_{}_{}-{}'.format(archive,
    #                                                                                                      1974, 1984),
    #                            distype='norm', topKneighbs=3)

    for archive in ['nahar', 'hayat']:
        if archive == 'nahar':
            get_changing_attitudes(word_lists1=target_list1, word_lists2=target_list2,
                                   neutral_lists=neutral_lists, archive=archive, desired_years=(1940, 1950),
                                   wemb_path=archives_wordembeddings,
                                   output_dir='latest', fig_name='israel_palestine_attitude_{}_{}-{}'.format(archive,
                                                                                                             1940, 1950),
                                   distype='norm', topKneighbs=3)

        get_changing_attitudes(word_lists1=target_list1, word_lists2=target_list2,
                               neutral_lists=neutral_lists, archive=archive, desired_years=(1950, 1960),
                               wemb_path=archives_wordembeddings,
                               output_dir='latest', fig_name='israel_palestine_attitude_{}_{}-{}'.format(archive,
                                                                                                         1950, 1960),
                               distype='norm', topKneighbs=3)

        get_changing_attitudes(word_lists1=target_list1, word_lists2=target_list2,
                               neutral_lists=neutral_lists, archive=archive, desired_years=(1960, 1970),
                               wemb_path=archives_wordembeddings,
                               output_dir='latest', fig_name='israel_palestine_attitude_{}_{}-{}'.format(archive,
                                                                                                         1960, 1970),
                               distype='norm', topKneighbs=3)

        get_changing_attitudes(word_lists1=target_list1, word_lists2=target_list2,
                               neutral_lists=neutral_lists, archive=archive, desired_years=(1974, 1984),
                               wemb_path=archives_wordembeddings,
                               output_dir='latest', fig_name='israel_palestine_attitude_{}_{}-{}'.format(archive,
                                                                                                         1974, 1984),
                               distype='norm', topKneighbs=3)

    for archive in ['assafir']:
        get_changing_attitudes(word_lists1=target_list1, word_lists2=target_list2,
                               neutral_lists=neutral_lists, archive=archive, desired_years=(1974, 1984),
                               wemb_path=archives_wordembeddings,
                               output_dir='latest', fig_name='israel_palestine_attitude_{}_{}-{}'.format(archive,
                                                                                                         1974, 1984),
                               distype='norm', topKneighbs=3)

    # for archive in ['assafir']:
    #     get_changing_attitudes(word_lists1=target_list1, word_lists2=target_list2,
    #                            neutral_lists=neutral_lists, archive=archive, desired_years=(1974, 1984),
    #                            wemb_path=archives_wordembeddings,
    #                            output_dir='latest', fig_name='israel_palestine_attitude_{}'.format(archive),
    #                            distype='norm', topKneighbs=3)


        # get_embedding_bias(word_list1=peace_practices, word_list2=occupation_practices, neutral_list=israel_list,
        #                    s_year=1933, e_year=1990, distype='cossim',
        #                    word2vec_models_path='E:/newspapers/word2vec/{}/embeddings/'.format(archive),
        #                    fig_name='{}_israel_palestine'.format(archive),
        #                    output_folder=output_dir,
        #                    ylab='Peace Practices')

        # get_embedding_bias_decade_level(word_list1=peace_practices,
        #                                 word_list2=occupation_practices,
        #                                 neutral_list=israel_list,
        #                                 decades_path='E:/newspapers/word2vec_decades/{}/meta_data/'.format(archive),
        #                                 archive=archive,
        #                                 fig_name='{}_israel_palestine_hiyam_decade'.format(archive),
        #                                 output_folder=output_dir,
        #                                 ylab='Peace Practices',
        #                                 distype='cossim')

    # else:
    #     get_embedding_bias(word_list1=peace_practices, word_list2=occupation_practices, neutral_list=israel_list,
    #                        s_year=1933, e_year=1990, distype='cossim',
    #                        word2vec_models_path='D:/word2vec/{}/embeddings/'.format(archive),
    #                        fig_name='{}_israel_palestine_hiyam'.format(archive),
    #                        output_folder=output_dir,
    #                        ylab='Peace Practices')
    #
    #     get_embedding_bias_decade_level(word_list1=peace_practices,
    #                                     word_list2=occupation_practices,
    #                                     neutral_list=israel_list,
    #                                     decades_path='D:/word2vec_decades/{}/meta_data/'.format(archive),
    #                                     archive=archive,
    #                                     fig_name='{}_israel_palestine_hiyam_decade'.format(archive),
    #                                     output_folder=output_dir,
    #                                     ylab='Peace Practices',
    #                                     distype='cossim')
