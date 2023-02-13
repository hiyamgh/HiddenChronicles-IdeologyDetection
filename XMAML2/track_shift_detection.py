import os
import pickle5 as pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

path = 'results/'
# DISCOURSES = ['ARG', 'VDC', 'PTC']
# for DISC in DISCOURSES:
#     fig, ax = plt.subplots(5, 5)
#     n = 0

for file in os.listdir(path):
    if '.pickle' in file:
        with open(os.path.join(path, file), 'rb') as handle:
            actual_predicted_years = pickle.load(handle)

        for model in actual_predicted_years:
            df_actual_predicted = actual_predicted_years[model]
            ap_counts_years = {}

            # nb of labels vs nb of years
            data_actual = np.zeros((len(set(df_actual_predicted['actual'])), len(set(df_actual_predicted['year']))))
            data_predicted = np.zeros((len(set(df_actual_predicted['actual'])), len(set(df_actual_predicted['year']))))

            for i, row in df_actual_predicted.iterrows():
                la = row['actual']
                lp = row['predicted']
                y = row['year']

                if y not in ap_counts_years:
                    ap_counts_years[y] = {}
                    ap_counts_years[y]['actual'] = {}
                    ap_counts_years[y]['predicted'] = {}

                if la not in ap_counts_years[y]['actual']:
                    ap_counts_years[y]['actual'][la] = 1
                else:
                    ap_counts_years[y]['actual'][la] += 1

                if lp not in ap_counts_years[y]['predicted']:
                    ap_counts_years[y]['predicted'][lp] = 1
                else:
                    ap_counts_years[y]['predicted'][lp] += 1

            years = list(ap_counts_years.keys())
            labels = list(set(df_actual_predicted['actual']))
            colors = get_cmap(n=len(labels))

            for j, y in enumerate(years):  # j
                for i, label in enumerate(labels):
                    data_actual[i, j] = ap_counts_years[y]['actual'][label] if label in ap_counts_years[y]['actual'] else 0
                    data_predicted[i, j] = ap_counts_years[y]['predicted'][label] if label in ap_counts_years[y]['predicted'] else 0

            with sns.axes_style("white"):
                sns.set_style("ticks")
                sns.set_context("talk")

                # plot details
                bar_width = 0.35
                epsilon = .015
                line_width = 1
                opacity = 0.7
                pos_bar_positions = np.arange(len(years))
                neg_bar_positions = pos_bar_positions + bar_width

                for i in range(data_actual.shape[0]):
                    if i == 0:
                        plt.bar(pos_bar_positions, data_actual[i, :], bar_width,
                                color=colors(i),
                                label=labels[i])

                        plt.bar(neg_bar_positions, data_predicted[i, :], bar_width,
                                color=colors(i),
                                hatch='//')
                        # label=labels[i])
                    else:
                        plt.bar(pos_bar_positions, data_actual[i, :], bar_width,
                                color=colors(i),
                                label=labels[i],
                                bottom=np.sum(data_actual[:i], axis=0))
                        plt.bar(neg_bar_positions, data_predicted[i, :], bar_width,
                                color=colors(i),
                                hatch='//',
                                # label=labels[i],
                                bottom=np.sum(data_predicted[:i], axis=0))

                plt.xticks(neg_bar_positions, years, rotation=45)
                plt.ylabel('Number of Labels')
                plt.legend(loc='best')
                plt.title(model)
                # plt.legend(bbox_to_anchor=(1.1, 1.05))
                sns.despine()

                fig = plt.gcf()
                fig.set_size_inches(12.5, 7.5)
                plt.tight_layout()

                fig_name = str(model).replace(' | few shot', '_few_shot')
                fig_name = fig_name.replace(' | zero shot', '_zero_shot')
                pref = file[:-7]

                plt.savefig(os.path.join(path, '{}_{}.png'.format(fig_name, pref)), dpi=300)
                plt.close()
