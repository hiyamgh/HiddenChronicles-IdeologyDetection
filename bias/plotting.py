import plotly.graph_objects as go
import os
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
sns.set(style="whitegrid") #TODO test this whitegrid, otherwise remove
import matplotlib.pyplot as plt


def mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# def plot_overtime_scatter(wl1, wl2, nl, arch, years, ):
#     arch = 'nahar'
#     biases_diff, casualties_diff = [], []
#     for year in range(1988, 2012):
#         emb_bias = get_embedding_bias_by_year(word_list1=participants_israel, word_list2=participants_palestine,
#                                               neutral_list=terrorism_list,
#                                               archive=arch,
#                                               year=year,
#                                               distype='cossim',
#                                               wemb_path=archives_wordembeddings[arch])
#         cas_diff = get_casualties_diff_by_year(year)
#         biases_diff.append(emb_bias)
#         casualties_diff.append(cas_diff)
#
#         print('year: {}, embedding_bias: {}'.format(year, emb_bias))


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


def plot_embedding_bias_time(embedding_biases, output_dir, fig_names, ylabs):
    mkdir(output_dir)
    min_year, max_year = -1, -1
    for archive in embedding_biases:
        min_year = min(embedding_biases[archive][0]['years'])
        max_year = max(embedding_biases[archive][0]['years'])
        break
    all_years = list(range(min_year, max_year + 1))

    for i in range(len(fig_names)):
        # fig = go.Figure()
        # get the ith data from each archive
        for archive in embedding_biases:
            bias = embedding_biases[archive][i]['biases']
            plt.plot(all_years, bias, label='{} bias'.format(archive))
            # fig.add_trace(go.Scatter(
            #     x=all_years,
            #     y=embedding_biases[archive][i]['biases'],
            #     name='<b>{}</b> bias'.format(archive), # Style name/legend entry with html tags
            #     connectgaps=True,# override default to connect the gaps
            #     mode='lines+markers'
            # ))

        # fig.update_layout(width=1500,
        #                   xaxis_title='Years',
        #                   yaxis_title=ylabs[i])
        # fig.write_image((os.path.join(output_dir, '{}.png'.format(fig_names[i]))))
        # fig.write_html((os.path.join(output_dir, '{}.html'.format(fig_names[i]))))
        plt.xlabel('Years')
        plt.ylabel(ylabs[i])
        plt.legend(loc='best')
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        fig.savefig(os.path.join(output_dir, '{}.png'.format(fig_names[i])))
        plt.close()

# def plot_embedding_bias_time(embedding_biases, output_dir, fig_names, ylabs):
#     mkdir(output_dir)
#     min_year, max_year = -1, -1
#     for archive in embedding_biases:
#         min_year = min(embedding_biases[archive][0]['years'])
#         max_year = max(embedding_biases[archive][0]['years'])
#         break
#     all_years = list(range(min_year, max_year + 1))
#
#     for i in range(len(fig_names)):
#         fig = go.Figure()
#         # get the ith data from each archive
#         for archive in embedding_biases:
#             fig.add_trace(go.Scatter(
#                 x=all_years,
#                 y=embedding_biases[archive][i]['biases'],
#                 name='<b>{}</b> bias'.format(archive), # Style name/legend entry with html tags
#                 connectgaps=True,# override default to connect the gaps
#                 mode='lines+markers'
#             ))
#
#         fig.update_layout(width=1500,
#                           xaxis_title='Years',
#                           yaxis_title=ylabs[i])
#         fig.write_image((os.path.join(output_dir, '{}.png'.format(fig_names[i]))))
#         # fig.write_html((os.path.join(output_dir, '{}.html'.format(fig_names[i]))))


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
