import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from bias.utilities import *
from scipy.stats.stats import pearsonr
import plotly.graph_objs as go
import plotly.figure_factory as ff

alphabet = list(range(0x0621, 0x063B)) + list(range(0x0641, 0x064B))
diactitics = list(range(0x064B, 0x0653))

alphabet = [chr(x) for x in alphabet]
diactitics = [chr(x) for x in diactitics]

# 1990 1991 1992 1993 1994 1995 1996 1997 1998 1999 2000

'''
target1 and target2 must be of equal size
attr1 and attr2 must be of equal size
apply the data augmentation I've been doing so far
'''

casualties = pd.read_csv('casualties/casualties_1988_2011.csv')

def get_embedding_bias_decade_level(word_list1, word_list2, neutral_list, decades_path,
                                    archive, fig_name, ylab, output_folder, distype='norm', topKneighbs=3):
    embedding_biases = []

    decades = get_decades(decades_path=decades_path)
    decades = sorted(decades, key = lambda x: x[0])
    for decade in decades:
        s_year = decade[0]
        e_year = decade[1]

        word2vec_currmodel = load_model_decade_level(decades_path, s_year, e_year)

        model_vocab = word2vec_currmodel.wv
        # word_list1_populated = list(set(populate_list(word_list1, word2vec_currmodel, model_vocab, topK=3)))
        # word_list2_populated = list(set(populate_list(word_list2, word2vec_currmodel, model_vocab, topK=3)))
        # neutral_list_populated = list(set(populate_list(neutral_list, word2vec_currmodel, model_vocab, topK=3)))
        #
        # print('list1 populated: {}'.format(word_list1_populated))
        # print('list2 populated: {}'.format(word_list2_populated))
        # print('neutral populated: {}'.format(neutral_list_populated))

        # get the list of terms of those that are present in the vocabulary
        word_list1, _ = check_terms(word_list1, word2vec_currmodel)
        word_list2, _ = check_terms(word_list2, word2vec_currmodel)
        neutral_list, _ = check_terms(neutral_list, word2vec_currmodel)

        neutral_list_populated = list(set(populate_list(neutral_list, word2vec_currmodel, model_vocab, topK=3)))

        if word_list1 == [] or word_list2 == [] or neutral_list == []:
            raise ValueError('one of the lists is empty')

        print('checked terms ...')

        if word_list1 and word_list2:
            if len(word_list1) == word_list2:
                pass
            elif len(word_list1) < len(word_list2):
                word_list1 = apply_augmentation(larger_list=word_list2, smaller_list=word_list1,
                                                word2vec_currmodel=word2vec_currmodel, topK=topKneighbs)
            else:
                word_list2 = apply_augmentation(larger_list=word_list1, smaller_list=word_list2,
                                                word2vec_currmodel=word2vec_currmodel, topK=topKneighbs)

        # get the representative vector from each group, which is the mean of the vectors of each word in the wordlist
        # repres1 = get_mean_vector(word2vec_currmodel, word_list1_populated)
        # repres2 = get_mean_vector(word2vec_currmodel, word_list2_populated)
        repres1 = get_mean_vector(word2vec_currmodel, word_list1)
        repres2 = get_mean_vector(word2vec_currmodel, word_list2)

        dist1 = np.mean([calc_distance_between_vectors(repres1, word2vec_currmodel.wv[w], distype=distype) for w in
                         neutral_list_populated])
        dist2 = np.mean([calc_distance_between_vectors(repres2, word2vec_currmodel.wv[w], distype=distype) for w in
                         neutral_list_populated])

        # embedding_bias = euclid1 - euclid2
        embedding_bias = dist1 - dist2
        print('----------------------------------- Decade: {}-{} -----------------------------------'.format(s_year, e_year))
        # print('distance females: {}'.format(dist1))
        # print('distance males: {}'.format(dist2))
        print('list1 populated: {}'.format(word_list1))
        print('list2 populated: {}'.format(word_list2))
        print('neutral populated: {}'.format(neutral_list_populated))
        print('Embedding Bias: {}'.format(dist1 - dist2))
        embedding_biases.append(embedding_bias)

    list_years = [d[0] for d in decades]
    biases = pd.DataFrame({'year': list_years, 'embedding_bias': embedding_biases})
    sns.lineplot(data=biases, x="year", y="embedding_bias", markers=True)

    # plt.plot(list(range(s_year, e_year + 1)), embedding_biases, marker='o', color='b')
    plt.xlabel('Years')
    plt.ylabel('Avg. Embedding Bias {}'.format(ylab))
    mkdir(directory=output_folder)
    plt.savefig(os.path.join(output_folder, '{}.png'.format(fig_name)))
    plt.close()


# compute Euclidean distance between representative and each word in the neutral word list of interest
# def get_embedding_bias(word_list1, word_list2, neutral_list, desired_archives, wemb_path,
#                        distype='norm', topKneighbs=3):

def calculate_embedding_bias(word_list1, word_list2, neutral_list, word2vec_currmodel, distype='norm'):
    ''' calculates the value of embedding bias, given target lists and attribute/neutral list '''
    repres1 = get_mean_vector(word2vec_currmodel, word_list1)
    repres2 = get_mean_vector(word2vec_currmodel, word_list2)

    dist1 = np.mean([calc_distance_between_vectors(repres1, word2vec_currmodel.wv[w], distype=distype) for w in neutral_list])
    dist2 = np.mean([calc_distance_between_vectors(repres2, word2vec_currmodel.wv[w], distype=distype) for w in neutral_list])

    embedding_bias = dist1 - dist2

    return embedding_bias


def get_casualties_diff_by_year(year):
    ''' census data difference (cannot get the percentage)'''
    if year in list(casualties['Year']):
        cas_israeli = int(casualties.loc[casualties.Year == year, 'Israelis'])
        cas_palestinians = int(casualties.loc[casualties.Year == year, 'Palestinians'])
        return cas_israeli - cas_palestinians


def get_embedding_bias_by_year(word_list1, word_list2, neutral_list, archive, year, wemb_path,
                               distype='norm', topKneighbs=3):
    ''' calculate the embedding bias per year '''
    word2vec_model = load_model_by_year(archive_path=wemb_path[archive], target_year=year)

    word_list1_found, word_list2_found, neutral_list_found = get_modified_word_lists(word_list1, word_list2,
                                                                                     neutral_list,
                                                                                     word2vec_model,
                                                                                     topKneighbs)
    print('{} - YEAR: {}'.format(archive, year))
    if word_list1_found == -1 or word_list2_found == -1 or neutral_list_found == -1:
        raise ValueError('could not find desired words in archive {} for year {}'.format(archive, year))
    else:
        print('list1 populated: {}'.format(word_list1_found))
        print('list2 populated: {}'.format(word_list2_found))
        print('neutral populated: {}'.format(neutral_list_found))

        embedding_bias = calculate_embedding_bias(word_list1_found, word_list2_found,
                                                  neutral_list_found, word2vec_model)
        return embedding_bias


def get_embedding_bias(word_lists1, word_lists2, neutral_lists, desired_archives, wemb_path,
                           distype='norm', topKneighbs=3):
    """
    wemb_path: dictionary mapping archive name to the path to its word embeddings
    :return:
    """
    embedding_biases = {}
    # get years total coverage (min_year_among_all_archives ==> max_year_among_all_archives)
    min_year, max_year = get_min_max_years(desired_archives, wemb_path)
    all_years = list(range(min_year, max_year + 1))

    for archive in desired_archives:
        print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ARCHIVE: {} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~'.format(archive))
        embedding_biases[archive] = {}
        word2vec_models_dict = load_all_models(archive_path=wemb_path[archive])
        archive_years = list(word2vec_models_dict.keys())

        for i in range(len(word_lists1)):

            embedding_biases[archive][i] = {}
            embedding_biases[archive][i]['years'] = []
            embedding_biases[archive][i]['biases'] = []

            # define current target-neutral lists
            word_list1 = word_lists1[i]
            word_list2 = word_lists2[i]
            neutral_list = neutral_lists[i]

            for year in all_years:
                print('----------------------------------- YEAR: {} -----------------------------------'.format(year))
                if year not in archive_years:
                    # check if key exists
                    print('year {} does not exist in archive {}'.format(year, archive))
                    embedding_biases[archive][i]['years'].append(year)
                    embedding_biases[archive][i]['biases'].append(None)

                else:
                    word2vec_currmodel = word2vec_models_dict[year]

                    word_list1_found, word_list2_found, neutral_list_found = get_modified_word_lists(word_list1, word_list2,
                                                                                                   neutral_list,
                                                                                                   word2vec_currmodel,
                                                                                                   topKneighbs)

                    if word_list1_found == -1 or word_list2_found == -1 or neutral_list_found == -1:
                        print('data augmentation with edit distances did not work')
                        embedding_biases[archive][i]['years'].append(year)
                        embedding_biases[archive][i]['biases'].append(None)
                        continue
                    else:
                        print('list1 populated: {}'.format(word_list1_found))
                        print('list2 populated: {}'.format(word_list2_found))
                        print('neutral populated: {}'.format(neutral_list_found))

                        # # get the representative vector from each group, which is the mean of the vectors of each word in the wordlist
                        # repres1 = get_mean_vector(word2vec_currmodel, word_list1_found)
                        # repres2 = get_mean_vector(word2vec_currmodel, word_list2_found)
                        #
                        # dist1 = np.mean(
                        #     [calc_distance_between_vectors(repres1, word2vec_currmodel.wv[w], distype=distype) for w in
                        #      neutral_list_found])
                        # dist2 = np.mean(
                        #     [calc_distance_between_vectors(repres2, word2vec_currmodel.wv[w], distype=distype) for w in
                        #      neutral_list_found])
                        #
                        # # embedding_bias = euclid1 - euclid2
                        # embedding_bias = dist1 - dist2

                        embedding_bias = calculate_embedding_bias(word_list1_found, word_list2_found, neutral_list_found, word2vec_currmodel)
                        print('Embedding Bias: {}'.format(embedding_bias))
                        embedding_biases[archive][i]['years'].append(year)
                        embedding_biases[archive][i]['biases'].append(embedding_bias)

    return embedding_biases

                    # word_list1_found, _ = check_terms(word_list1, word2vec_currmodel)
                    # word_list2_found, _ = check_terms(word_list2, word2vec_currmodel)
                    # neutral_list_found, _ = check_terms(neutral_list, word2vec_currmodel)
                    #
                    # if word_list1_found == [] or word_list2_found == [] or neutral_list_found == []:
                    #     if word_list1_found == []:
                    #         word_list1_found = get_possibilities(word_list1_found, word2vec_currmodel.wv)
                    #     if word_list2_found == []:
                    #         word_list2_found = get_possibilities(word_list2_found, word2vec_currmodel.wv)
                    #     if neutral_list_found == []:
                    #         neutral_list_found = get_possibilities(neutral_list_found, word2vec_currmodel.wv)
                    #
                    #     if word_list1_found == [] or word_list2_found == [] or neutral_list_found == []:
                    #         print('data augmentation with edit distances did not work')
                    #         embedding_biases[archive][i]['years'].append(year)
                    #         embedding_biases[archive][i]['biases'].append(None)
                    #         continue
                    #
                    # print('checked terms ...')
                    #
                    # if word_list1_found and word_list2_found:
                    #     if len(word_list1_found) == len(word_list2_found):
                    #         pass
                    #     elif len(word_list1_found) < len(word_list2_found):
                    #         word_list1_found = apply_augmentation(larger_list=word_list2_found, smaller_list=word_list1_found,
                    #                                         word2vec_currmodel=word2vec_currmodel, topK=topKneighbs)
                    #     else:
                    #         word_list2_found = apply_augmentation(larger_list=word_list1_found, smaller_list=word_list2_found,
                    #                                         word2vec_currmodel=word2vec_currmodel, topK=topKneighbs)
                    #
                    # model_vocab = word2vec_currmodel.wv
                    #
                    # neutral_list_populated = list(set(populate_list(neutral_list_found, word2vec_currmodel, model_vocab, topK=3)))

                    # print('----------------------------------- YEAR: {} -----------------------------------'.format(year))
                    #
                    # print('list1 populated: {}'.format(word_list1_found))
                    # print('list2 populated: {}'.format(word_list2_found))
                    # print('neutral populated: {}'.format(neutral_list_populated))
                    #
                    # # get the representative vector from each group, which is the mean of the vectors of each word in the wordlist
                    # repres1 = get_mean_vector(word2vec_currmodel, word_list1_found)
                    # repres2 = get_mean_vector(word2vec_currmodel, word_list2_found)
                    #
                    # dist1 = np.mean(
                    #     [calc_distance_between_vectors(repres1, word2vec_currmodel.wv[w], distype=distype) for w in
                    #      neutral_list_populated])
                    # dist2 = np.mean(
                    #     [calc_distance_between_vectors(repres2, word2vec_currmodel.wv[w], distype=distype) for w in
                    #      neutral_list_populated])
                    #
                    # # embedding_bias = euclid1 - euclid2
                    # embedding_bias = dist1 - dist2
                    # print('Embedding Bias: {}'.format(dist1 - dist2))
                    # embedding_biases[archive][i]['years'].append(year)
                    # embedding_biases[archive][i]['biases'].append(embedding_bias)

    # return embedding_biases


def get_modified_word_lists(word_list1, word_list2, neutral_list, word2vec_currmodel, topKneighbs):
    word_list1, _ = check_terms(word_list1, word2vec_currmodel)
    word_list2, _ = check_terms(word_list2, word2vec_currmodel)
    neutral_list, _ = check_terms(neutral_list, word2vec_currmodel)

    if word_list1 == [] or word_list2 == [] or neutral_list == []:
        if not word_list1:
            word_list1 = get_possibilities(word_list1, word2vec_currmodel.wv)
        if not word_list2:
            word_list2 = get_possibilities(word_list2, word2vec_currmodel.wv)
        if not neutral_list:
            neutral_list = get_possibilities(neutral_list, word2vec_currmodel.wv)

        if word_list1 == [] or word_list2 == [] or neutral_list == []:
            return -1, -1, -1

    if word_list1 and word_list2 and neutral_list:
        if len(word_list1) == len(word_list2):
            pass
        elif len(word_list1) < len(word_list2):
            word_list1 = apply_augmentation(larger_list=word_list2, smaller_list=word_list1,
                                                  word2vec_currmodel=word2vec_currmodel, topK=topKneighbs)
        else:
            word_list2 = apply_augmentation(larger_list=word_list1, smaller_list=word_list2,
                                                  word2vec_currmodel=word2vec_currmodel, topK=topKneighbs)

    model_vocab = word2vec_currmodel.wv
    neutral_list = list(set(populate_list(neutral_list, word2vec_currmodel, model_vocab, topK=3)))

    return word_list1, word_list2, neutral_list


def get_changing_attitudes(word_lists1, word_lists2, neutral_lists, archive, desired_years,
                           wemb_path, output_dir, fig_name,
                           distype='norm', topKneighbs=3):
    '''
        desired_years: tuple of two years, start and end
        wemb_path: dictionary of paths to word embeddings folder in each archive
    '''
    # 0  0.14 0.15 0.16
    # 2 0.14 0.15 0.16
    # 3 0.14 0.15 0.16
    start_year, end_year = desired_years[0], desired_years[1]
    all_years = list(range(start_year, end_year + 1))

    # filter the years that are actually found in the archive
    archive_years = get_archive_year(archive=archive, archive_path=wemb_path)
    all_years = [y for y in all_years if y in archive_years]
    embedding_biases = {}
    for i in range(len(word_lists1)):
        print('{}st word list'.format(i))
        embedding_biases[i] = []
        word_list1 = word_lists1[i]
        word_list2 = word_lists2[i]
        neutral_list = neutral_lists[i]

        for year in all_years:
            word2vec_currmodel = Word2Vec.load(os.path.join(wemb_path[archive], 'word2vec_{}'.format(year)))
            word_list1_found, word_list2_found, neutral_list_found = get_modified_word_lists(word_list1, word_list2,
                                                                                             neutral_list,
                                                                                             word2vec_currmodel,
                                                                                             topKneighbs)

            embedding_bias = calculate_embedding_bias(word_list1_found, word_list2_found,
                                                      neutral_list_found, word2vec_currmodel,
                                                      distype=distype)
            embedding_biases[i].append(embedding_bias)

    heatmap = np.zeros((len(all_years), len(all_years)))
    heatmap_pvalues = np.zeros((len(all_years), len(all_years)))
    for p1 in range(len(all_years)):
        for p2 in range(len(all_years)):
            # heatmap[p1, p2], heatmap_pvalues[p1, p2] = pearsonr([embedding_biases[i][p1] for i in embedding_biases],
            #                                                     [embedding_biases[i][p2] for i in embedding_biases])

            value, p_value = pearsonr([embedding_biases[i][p1] for i in embedding_biases],
                                      [embedding_biases[i][p2] for i in embedding_biases])
            heatmap[p1, p2] = float("{:.2f}".format(value))

    # dtype = np.int16
    layout_heatmap = go.Layout(
        xaxis=dict(title='Years'),
        yaxis=dict(title='Years'),
    )

    # all_years = [str(y) for y in all_years]
    ff_fig = ff.create_annotated_heatmap(x=all_years, y=all_years, z=heatmap.tolist(), showscale=True,
                                         colorscale='Viridis',)
    fig = go.FigureWidget(ff_fig)
    fig.layout = layout_heatmap
    fig.layout.annotations = ff_fig.layout.annotations
    fig['layout']['yaxis']['autorange'] = "reversed"

    # if archive == 'nahar':
    #     fig.add_shape(type="rect",
    #                   x0=1960, y0=1960, x1=1966, y1=1966,
    #                   line=dict(color="red"),
    #                   )
    #     fig.add_shape(type="rect",
    #                   x0=1967, y0=1967, x1=1970, y1=1970,
    #                   line=dict(color="red"),
    #                   )

    # if archive == 'hayat':
    #     fig.add_shape(type="rect",
    #                   x0=1960, y0=1960, x1=1966, y1=1966,
    #                   line=dict(color="red"),
    #                   )
    #     fig.add_shape(type="rect",
    #                   x0=1967, y0=1967, x1=1970, y1=1970,
    #                   line=dict(color="red"),
    #                   )
    # fig.update_xaxes(type='category')
    # fig.update_yaxes(type='category')
    # fig = ff.create_annotated_heatmap(heatmap, x=all_years, y=all_years, colorscale='Viridis', showscale = True)

    # if archive == 'nahar':
    #     fig.add_shape(type="rect",
    #                   x0=-0.5, y0=3.5, x1=6.5, y1=10.5,
    #                   line=dict(color="red", width=4))
    #     fig.add_shape(type="rect",
    #                   x0=6.5, y0=-0.5, x1=10.5, y1=3.5,
    #                   line=dict(color="red", width=4))
    #     fig.update_shapes(dict(xref='x', yref='y'))
    #
    # if archive == 'hayat':
    #     fig.add_shape(type="rect",
    #                   x0=-0.5, y0=4.5, x1=5.5, y1=10.5,
    #                   line=dict(color="red", width=4),
    #                   )
    #     fig.add_shape(type="rect",
    #                   x0=4.5, y0=-0.5, x1=4.5, y1=10.5,
    #                   line=dict(color="red", width=4),
    #                   )
    #     fig.update_shapes(dict(xref='x', yref='y'))

    mkdir(output_dir)
    fig.write_image(os.path.join(output_dir, '{}.png'.format(fig_name)))
    fig.write_html((os.path.join(output_dir, '{}.html'.format(fig_name))))

        # if word_list1_found == -1 or word_list2_found == -1 or neutral_list_found == -1:
    #         for en1 in range(len(yrs_to_include)):
    #         for en2 in range(len(yrs_to_include)):
    #             xrank = scipy.stats.stats.rankdata(difs_by_year[en1])
    #             yrank = scipy.stats.stats.rankdata(difs_by_year[en2])
    #             # heatmap[en1, en2], heatmap_pvalues[en1, en2]  = kendalltau(xrank, yrank)
    #             heatmap[en1, en2], heatmap_pvalues[en1, en2]  = pearsonr(difs_by_year[en1], difs_by_year[en2])




    # for archive in desired_archives:
    #     print('archive: {}'.format(archive))
    #     print('years: {}'.format(embedding_biases[archive]['years']))
    #     print('embedding_biases: {}'.format(embedding_biases[archive]['biases']))
    #
    # all_embedding_biases[i] = embedding_biases

        # return embedding_biases
    # return all_embedding_biases

                # embedding_biases[archive]['years'] = embedding_bias
                # embedding_bias_year[year] = embedding_bias

            # list_years = list(range(s_year, e_year + 1))
            # biases = pd.DataFrame({'year': list_years, 'embedding_bias': embedding_biases})
            # sns.lineplot(data=biases, x="year", y="embedding_bias", markers=True)
            #
            # # plt.plot(list(range(s_year, e_year + 1)), embedding_biases, marker='o', color='b')
            # plt.xlabel('Years')
            # plt.ylabel('Avg. Embedding Bias {}'.format(ylab))
            # mkdir(directory=output_folder)
            # plt.savefig(os.path.join(output_folder, '{}.png'.format(fig_name)))
            # plt.close()


def populate_list(target_terms, model, vocab, topK):
    populated = []
    for t in target_terms:
        if t in vocab:
            populated.append(t)
            tedit1 = edits1(t)
            tedit1 = [te for te in tedit1 if te in vocab]

            tedit1sorted = {}
            for te in tedit1:
                tedit1sorted[te] = model.similarity(t, te)
            tedit1sorted = {k: v for k, v in sorted(tedit1sorted.items(), key=lambda item: item[1], reverse=True)}
            top_edits = dict(islice(tedit1sorted.items(), topK))
            # print('\n top neighbors for {}:'.format(t))
            # for k, v in top_edits.items():
            #     print('{}: {}'.format(k, v))
            populated = populated + list(top_edits.keys())
        else:
            possible_from_missing = get_edits_missing(t, vocab)
            if possible_from_missing != -1:
                # print('\n top neighbors for {} (did not find t, getting its edits that exist in vocab):'.format(t))
                # print('edits that exist: {}'.format(possible_from_missing))
                populated = populated + possible_from_missing
            else:
                # print('skipping {} because neither it nor its edits are in the vocab'.format(t))
                pass

    return populated
