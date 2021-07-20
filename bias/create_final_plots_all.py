from embedding_biases import *
from plotting import *

archives_wordembeddings = {
        'nahar': '../input/word2vec/nahar/embeddings/',
        'hayat': '../input/word2vec/hayat/embeddings/',
        'assafir': '../input/word2vec/assafir/embeddings/',
    }

keyword_paths = {

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


def plot_overtime_scatter(word_list1, word_list2, neutral_list, years, archive, topK,
                          ylab, fig_name, out_folder):
    years = list(sorted(years))
    biases_cas = []
    for yr in years:
        emb_bias = get_embedding_bias_by_year(word_list1, word_list2, neutral_list,
                                              archive, yr, archives_wordembeddings, topKneighbs=topK)
        casualties = get_casualties_diff_by_year(yr)
        biases_cas.append((emb_bias, casualties))

    plot_bias_overtime_scatter_casualties(biases_cass=biases_cas, ylab=ylab,
                                          fig_name=fig_name, out_folder=out_folder)


if __name__ == '__main__':

    # target list 1 vs. target list 2 - neutral list
    # peace vs. occupation - israel
    # participant israel vs. participant palestine - occupation
    # participant israel vs. participant palestine - terrorism
    # participant israel vs. participant palestine - methods of violence

    # get the keywords
    main_dir = 'israeli_palestinian_conflict/occupations_vs_peace+israel/'
    peace_practices = file_to_list(txt_file=os.path.join(main_dir, 'non_occupation_practices_arabic.txt'))
    # occupation_practices = file_to_list(txt_file=os.path.join(main_dir, 'occupation_practices_arabic.txt'))
    israel_list = file_to_list(txt_file=os.path.join(main_dir, 'israel_list_arabic.txt'))
    palestine_list = file_to_list(txt_file=os.path.join(main_dir, 'participants_palestine_arabic.txt'))

    # set 2
    # participants israel vs. participants palestine
    main_dir2 = 'israeli_palestinian_conflict/participants+methods_violence/'
    participants_israel = file_to_list(txt_file=os.path.join(main_dir2, 'participants_Israel_arabic.txt'))
    participants_palestine = file_to_list(txt_file=os.path.join(main_dir2, 'participants_palestine_arabic.txt'))
    # occupation_list = file_to_list(txt_file='occupation/occupation_list_arabic.txt')

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
    neutral_lists = [terrorism_list, methods_of_violence, peace_practices]

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

    plot_overtime_scatter(word_list1=participants_israel, word_list2=participants_palestine,
                          neutral_list=terrorism_list, years=list(range(1988, 2012)),
                          archive='assafir', topK=3, ylab='Israeli Terrorism-Bias',
                          fig_name='assafir_bias_casualties_terrorism', out_folder='latest/census/')

    plot_overtime_scatter(word_list1=participants_israel, word_list2=participants_palestine,
                          neutral_list=methods_of_violence, years=list(range(1988, 2012)),
                          archive='assafir', topK=3, ylab='Israeli Violence-Bias',
                          fig_name='assafir_bias_casualties_violence', out_folder='latest/census/')

    plot_overtime_scatter(word_list1=participants_israel, word_list2=participants_palestine,
                          neutral_list=peace_practices, years=list(range(1988, 2012)),
                          archive='assafir', topK=3, ylab='Israeli Peace-Bias',
                          fig_name='assafir_bias_casualties_peace', out_folder='latest/census/')


    # for archive in ['nahar', 'hayat']:
    #     if archive == 'nahar':
    #         get_changing_attitudes(word_lists1=target_list1, word_lists2=target_list2,
    #                                neutral_lists=neutral_lists, archive=archive, desired_years=(1940, 1950),
    #                                wemb_path=archives_wordembeddings,
    #                                output_dir='latest', fig_name='israel_palestine_attitude_{}_{}-{}'.format(archive,
    #                                                                                                          1940, 1950),
    #                                distype='norm', topKneighbs=3)
    #
    #     get_changing_attitudes(word_lists1=target_list1, word_lists2=target_list2,
    #                            neutral_lists=neutral_lists, archive=archive, desired_years=(1950, 1960),
    #                            wemb_path=archives_wordembeddings,
    #                            output_dir='latest', fig_name='israel_palestine_attitude_{}_{}-{}'.format(archive,
    #                                                                                                      1950, 1960),
    #                            distype='norm', topKneighbs=3)
    #
    #     get_changing_attitudes(word_lists1=target_list1, word_lists2=target_list2,
    #                            neutral_lists=neutral_lists, archive=archive, desired_years=(1960, 1970),
    #                            wemb_path=archives_wordembeddings,
    #                            output_dir='latest', fig_name='israel_palestine_attitude_{}_{}-{}'.format(archive,
    #                                                                                                      1960, 1970),
    #                            distype='norm', topKneighbs=3)
    #
    #     get_changing_attitudes(word_lists1=target_list1, word_lists2=target_list2,
    #                            neutral_lists=neutral_lists, archive=archive, desired_years=(1974, 1984),
    #                            wemb_path=archives_wordembeddings,
    #                            output_dir='latest', fig_name='israel_palestine_attitude_{}_{}-{}'.format(archive,
    #                                                                                                      1974, 1984),
    #                            distype='norm', topKneighbs=3)
    #
    # for archive in ['assafir']:
    #     get_changing_attitudes(word_lists1=target_list1, word_lists2=target_list2,
    #                            neutral_lists=neutral_lists, archive=archive, desired_years=(1974, 1984),
    #                            wemb_path=archives_wordembeddings,
    #                            output_dir='latest', fig_name='israel_palestine_attitude_{}_{}-{}'.format(archive,
    #                                                                                                      1974, 1984),
    #                            distype='norm', topKneighbs=3)



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
