from embedding_biases import *
from plotting import *
import os
import pickle

archives_wordembeddings = {
    'nahar': '../input/word2vec/nahar/embeddings/',
    'hayat': '../input/word2vec/hayat/embeddings/',
    'assafir': '../input/word2vec/assafir/embeddings/',
}


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
    main_dir = 'keywords/arabic/'
    peace_practices = file_to_list(txt_file=os.path.join(main_dir, 'non_occupation_practices_arabic.txt'))
    israel_list = file_to_list(txt_file=os.path.join(main_dir, 'israel_list_arabic.txt'))
    palestine_list = file_to_list(txt_file=os.path.join(main_dir, 'participants_palestine_arabic.txt'))

    participants_israel = file_to_list(txt_file=os.path.join(main_dir, 'participants_Israel_arabic.txt'))
    participants_palestine = file_to_list(txt_file=os.path.join(main_dir, 'participants_palestine_arabic.txt'))
    # terrorism_list = file_to_list(txt_file='terrorism/terrorism_list_arabic.txt')
    terrorism_list = file_to_list(txt_file=os.path.join(main_dir, 'terrorism(100yearsofbias)_arabic.txt'))
    methods_of_violence = file_to_list(txt_file=os.path.join(main_dir, 'methods_of_violence_arabic.txt'))

    target_list1 = [participants_israel, participants_israel, participants_israel]
    target_list2 = [participants_palestine, participants_palestine, participants_palestine]
    neutral_lists = [terrorism_list, methods_of_violence, peace_practices]

    with open('embedding_biases.pickle', 'rb') as handle:
        all_embedding_biases = pickle.load(handle)

    plots_to_do_scatter_casualties = [
        [plot_overtime_scatter, [participants_israel, participants_palestine, terrorism_list, list(range(1988, 2012)), 'assafir', 3, 'Israeli-Terrorism Bias', 'assafir_israeli_terrorism', 'output/scatter/']],
        [plot_overtime_scatter, [participants_israel, participants_palestine, methods_of_violence, list(range(1988, 2012)), 'assafir', 3, 'Israeli-Violence Bias', 'assafir_israeli_violence', 'output/scatter/']],
        [plot_overtime_scatter, [participants_israel, participants_palestine, peace_practices, list(range(1988, 2012)), 'assafir', 3, 'Israeli-Peace Bias', 'assafir_israeli_peace', 'output/scatter/']],
    ]

    plots_to_do_bias_overtime = [
        [get_embedding_bias, [[participants_israel, participants_israel, participants_israel],
                              [participants_palestine, participants_palestine, participants_palestine],
                              [terrorism_list, methods_of_violence, peace_practices],
                              ['nahar', 'hayat', 'assafir'], archives_wordembeddings, 'cosine', 3]]
    ]

    plots_to_do_heatmap = [
        [cross_time_correlation_heatmap, [all_embedding_biases, 'nahar', 'output/heatmap/', 1933, 1943, 'cross_time_1933_1943']],
        [cross_time_correlation_heatmap, [all_embedding_biases, 'nahar', 'output/heatmap/', 1974, 1984, 'cross_time_1974_1984']],
        [cross_time_correlation_heatmap, [all_embedding_biases, 'nahar', 'output/heatmap/', 2000, 2009, 'cross_time_2000_2009']],

        # [cross_time_correlation_heatmap, [all_embedding_biases, 'hayat', 'output/heatmap/', 1974, 1984, 'cross_time_1974_1984']],
        # [cross_time_correlation_heatmap, [all_embedding_biases, 'assafir', 'output/heatmap/', 2000, 2009, 'cross_time_2000_2009']],

    ]

    plots_to_do_bias_casualties_overtime = [
        [plot_embedding_bias_census, [all_embedding_biases, 0, 'nahar', 1990, 2009, 'Avg. Embedding Bias Israeli Terrorism', 'output/bias_casualties_over_time/', 'israel_palestine_terrorism']],
        [plot_embedding_bias_census, [all_embedding_biases, 0, 'assafir', 1990, 2011, 'Avg. Embedding Bias Israeli Terrorism', 'output/bias_casualties_over_time/', 'israel_palestine_terrorism']],
        [plot_embedding_bias_census, [all_embedding_biases, 1, 'nahar', 1990, 2009, 'Avg. Embedding Bias Israeli Methods of Violence', 'output/bias_casualties_over_time/', 'israel_palestine_violence']],
        [plot_embedding_bias_census, [all_embedding_biases, 1, 'assafir', 1990, 2011, 'Avg. Embedding Bias Israeli Methods of Violence',  'output/bias_casualties_over_time/', 'israel_palestine_violence']],
        # [plot_embedding_bias_census, [all_embedding_biases, 2, 'nahar', 1990, 2009, 'Avg. Embedding Bias Israeli Peace Practices', 'output/bias_casualties_over_time/', 'israel_palestine_peace']],
        [plot_embedding_bias_census, [all_embedding_biases, 2, 'assafir', 1990, 2011, 'Avg. Embedding Bias Israeli Peace Practices', 'output/bias_casualties_over_time/', 'israel_palestine_peace']]
    ]

    plots_to_do_bias_start_end_overtime = [
        [plot_embedding_bias_start_end, [all_embedding_biases, 0, ['nahar', 'assafir'], 1974, 1990, 'Avg. Embedding Bias Israeli Terrorism', 'output/bias_overtime_start_end/', 'israel_palestine_terrorism']],
        [plot_embedding_bias_start_end, [all_embedding_biases, 1, ['nahar', 'assafir'], 1974, 1990, 'Avg. Embedding Bias Israeli Violence', 'output/bias_overtime_start_end/', 'israel_palestine_violence']],

        [plot_embedding_bias_start_end, [all_embedding_biases, 0, ['nahar', 'assafir', 'hayat'], 1950, 1960, 'Avg. Embedding Bias Israeli Terrorism', 'output/bias_overtime_start_end/', 'israel_palestine_terrorism']],
        [plot_embedding_bias_start_end, [all_embedding_biases, 1, ['nahar', 'assafir', 'hayat'], 1950, 1960, 'Avg. Embedding Bias Israeli Violence', 'output/bias_overtime_start_end/', 'israel_palestine_violence']],

        [plot_embedding_bias_start_end, [all_embedding_biases, 2, ['nahar', 'assafir'], 1990, 2000, 'Avg. Embedding Bias Israeli Peace', 'output/bias_overtime_start_end/', 'israel_palestine_peace']],

        [plot_embedding_bias_start_end, [all_embedding_biases, 0, ['nahar', 'assafir'], 2000, 2009, 'Avg. Embedding Bias Israeli Terrorism', 'output/bias_overtime_start_end/', 'israel_palestine_terrorism']],
        [plot_embedding_bias_start_end, [all_embedding_biases, 1, ['nahar', 'assafir'], 2000, 2009, 'Avg. Embedding Bias Israeli Violence', 'output/bias_overtime_start_end/', 'israel_palestine_violence']],

    ]

    fig_names = ['israel_palestine_terrorism', 'israel_palestine_violence', 'israel_palestine_peace']
    ylabs = ['Avg. Embedding Bias Israeli Terrorism', 'Avg. Embedding Bias Israeli Methods of Violence', 'Avg. Embedding Bias Peace Practices']


    for plot in plots_to_do_bias_overtime:
        # print(plot[0], plot[1][1:])
        # all_embedding_biases = plot[0](*plot[1])
        # with open('embedding_biases.pickle', 'wb') as handle:
        #     pickle.dump(all_embedding_biases, handle, protocol=pickle.HIGHEST_PROTOCOL)
        plot_embedding_bias_time(all_embedding_biases, 'output/bias_over_time/', fig_names, ylabs)

    for plot in plots_to_do_heatmap:
        print(plot[0], plot[1][1:])
        plot[0](*plot[1])
        print('==============================================================')

    for plot in plots_to_do_bias_casualties_overtime:
        print(plot[0], plot[1][1:])
        plot[0](*plot[1])
        print('==============================================================')

    for plot in plots_to_do_bias_start_end_overtime:
        print(plot[0], plot[1][1:])
        plot[0](*plot[1])
        print('==============================================================')