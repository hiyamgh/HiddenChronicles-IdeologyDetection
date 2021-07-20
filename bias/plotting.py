import plotly.graph_objects as go
import plotly.figure_factory as ff
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
sns.set(style="whitegrid")
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from scipy.signal import savgol_filter

casualties = pd.read_csv('casualties/casualties_1988_2011.csv')
casualties_dict = {}
for i, row in casualties.iterrows():
    year = int(row['Year'])
    israeli_cas = int(row['Israelis'])
    palestinian_cas = int(row['Palestinians'])

    casualties_dict[year] = {
        'israel': israeli_cas,
        'palestine': palestinian_cas
    }


def mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def plot_bias_overtime_scatter_casualties(biases_cass, ylab, fig_name, out_folder):
    ''' plot bias over time with scatter plot of regression of emb_boas vs casualties '''
    emb_biases = [eb for eb, _ in biases_cass]
    cas = [c for _, c in biases_cass] # for casualties
    print('{} & {} & {} & {} & {} \\\\'.format('r^2', 'coefficient p-value', 'coefficient value', 'intercept p-value', 'intercept value') )
    df = pd.DataFrame([emb_biases, cas])
    df = df.transpose()
    df.columns = ['embedding bias', 'cas num']
    df['const'] = 1
    model = sm.OLS(df['embedding bias'], df[['cas num', 'const']]).fit()
    print('${:.4}$ & ${:.4}$ & ${:.4} \pm {:.4}$& ${:.4}$ & ${:.4} \pm {:.4}$\\\\'.format(model.rsquared,model.pvalues[0], model.params[0], model.bse[0],model.pvalues[1], model.params[1], model.bse[1] )) # summarize_model(model)
    scatter_kws = {"s" : 20}
    # if years_all is not None:
    #     yrs_order = list(sorted(set(years_all)))
    #     pallete = sns.color_palette("hls", len(yrs_order))
    #     color = [pallete[yrs_order.index(y)] for y in yrs_order]
    #     scatter_kws['color']= color

    sns.regplot(x=cas, y=emb_biases, scatter = True, scatter_kws = scatter_kws, truncate  = True)#,scatter_kws={"s": sizes})
    sns.despine()
    plt.xlabel('Israeli Fatalities Raw Difference')
    plt.ylabel(ylab)
    mkdir(out_folder)
    plt.savefig(os.path.join(out_folder, '{}.png'.format(fig_name)))
    plt.close()

    with open(os.path.join(out_folder, '{}.txt'.format(fig_name)), 'w') as f:
        f.writelines('{} & {} & {} & {} & {} \\\\\n'.format('r^2', 'coefficient p-value', 'coefficient value', 'intercept p-value', 'intercept value'))
        f.writelines('${:.4}$ & ${:.4}$ & ${:.4} \pm {:.4}$& ${:.4}$ & ${:.4} \pm {:.4}$\\\\'.format(model.rsquared,model.pvalues[0], model.params[0], model.bse[0],model.pvalues[1], model.params[1], model.bse[1]))
        f.close()


def plot_embedding_bias_census(embedding_biases, bias_type, archive, start_year, end_year, ylab, output_folder, figname):
    all_years = list(range(int(start_year), int(end_year) + 1))
    years_available = embedding_biases[archive][bias_type]['years']
    biases, casualties_diff = [], []
    # get bias corresponding to each year of interest
    for i, year in enumerate(years_available):
        if int(year) in all_years:
            biases.append(embedding_biases[archive][bias_type]['biases'][i])

    biases_sm = savgol_filter(biases, 17, 3)

    for year in all_years:
        israeli_cas = casualties_dict[year]['israel']
        palestinian_cas = casualties_dict[year]['palestine']
        casualties_diff.append(israeli_cas - palestinian_cas)

    casualties_diff_sm = savgol_filter(casualties_diff, 17, 3)

    # some confidence interval
    ci_bias = 1.96 * np.std(biases) / np.mean(biases)
    ci_casualties = 1.96 * np.std(casualties_diff)/np.mean(casualties_diff)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(all_years, biases_sm, 'g-', marker='o')
    # ax1.fill_between(all_years, (biases_sm - ci_bias), (biases_sm + ci_bias), color='g', alpha=.1)
    ax2.plot(all_years, casualties_diff_sm, 'b-', marker='o')
    # ax2.fill_between(all_years, (casualties_diff_sm - ci_casualties), (casualties_diff_sm + ci_casualties), color='b', alpha=.1)

    ax1.set_xlabel('Years')
    ax1.set_ylabel(ylab, color='g')
    ax2.set_ylabel('Casualties difference', color='b')

    mkdir(output_folder)
    fig.set_size_inches(18.5, 10.5)
    fig.savefig(os.path.join(output_folder, '{}_{}'.format(figname, archive)))
    plt.close()
    # fig.close()


def plot_embedding_bias_time(embedding_biases, output_dir, fig_names, ylabs):
    mkdir(output_dir)
    min_year, max_year = -1, -1
    for archive in embedding_biases:
        min_year = min(embedding_biases[archive][0]['years'])
        max_year = max(embedding_biases[archive][0]['years'])
        break
    all_years = list(range(min_year, max_year + 1))

    for i in range(len(fig_names)):
        # get the ith data from each archive
        for archive in embedding_biases:
            bias = embedding_biases[archive][i]['biases']
            plt.plot(all_years, bias, label='{} bias'.format(archive), marker='o')

        plt.xlabel('Years')
        plt.ylabel(ylabs[i])
        plt.legend(loc='best')
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        fig.savefig(os.path.join(output_dir, '{}.png'.format(fig_names[i])))
        plt.close()


def cross_time_correlation_heatmap(embedding_biases, archive, output_dir, start_year, end_year, fig_name):

    all_years = list(range(start_year, end_year+1))
    heatmap = np.zeros((len(all_years), len(all_years)))
    heatmap_pvalues = np.zeros((len(all_years), len(all_years)))
    for p1 in range(len(all_years)):
        for p2 in range(len(all_years)):

            value, p_value = pearsonr([embedding_biases[archive][i]['biases'][p1] for i in range(len(embedding_biases))],
                                      [embedding_biases[archive][i]['biases'][p2] for i in range(len(embedding_biases))])
            heatmap[p1, p2] = float("{:.2f}".format(value))
            heatmap_pvalues[p1, p2] = p_value

            if p_value <= 0.05:
                print('{}, {} -> {}'.format(p1, p2, p_value))

    # dtype = np.int16
    layout_heatmap = go.Layout(
        xaxis=dict(title='Years'),
        yaxis=dict(title='Years'),
    )

    ff_fig = ff.create_annotated_heatmap(x=all_years, y=all_years, z=heatmap.tolist(), showscale=True,
                                         colorscale='Viridis',)
    fig = go.FigureWidget(ff_fig)
    fig.layout = layout_heatmap
    fig.layout.annotations = ff_fig.layout.annotations
    fig['layout']['yaxis']['autorange'] = "reversed"

    mkdir(output_dir)
    fig.write_image(os.path.join(output_dir, '{}_{}.png'.format(fig_name, archive)))


def plot_counts(counts_biases, output_dir, fig_name):
    mkdir(output_dir)
    min_year, max_year = -1, -1
    for archive in counts_biases:
        min_year = min(counts_biases[archive]['years'])
        max_year = max(counts_biases[archive]['years'])
        break
    all_years = list(range(min_year, max_year + 1))
    fig = go.Figure()
    for archive in counts_biases:
        fig.add_trace(go.Scatter(
            x=all_years,
            y=counts_biases[archive]['counts'],
            name='<b>{}</b>'.format(archive),  # Style name/legend entry with html tags
            connectgaps=True,  # override default to connect the gaps
            mode='lines+markers'
        ))
    fig.update_layout(width=1500,
                      xaxis_title='Years',
                      yaxis_title='Counts of Israeli-Related Words')
    fig.write_image((os.path.join(output_dir, '{}.png'.format(fig_name))))
    # fig.write_html((os.path.join(output_dir, '{}.html'.format(fig_name))))
