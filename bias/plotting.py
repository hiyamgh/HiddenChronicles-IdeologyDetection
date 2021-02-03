import plotly.graph_objects as go
import os


def mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def plot_embedding_bias_time(embedding_biases, output_dir, fig_names, ylabs):
    mkdir(output_dir)
    min_year, max_year = -1, -1
    for archive in embedding_biases:
        min_year = min(embedding_biases[archive][0]['years'])
        max_year = max(embedding_biases[archive][0]['years'])
        break
    all_years = list(range(min_year, max_year + 1))

    for i in range(len(fig_names)):
        fig = go.Figure()
        # get the ith data from each archive
        for archive in embedding_biases:
            fig.add_trace(go.Scatter(
                x=all_years,
                y=embedding_biases[archive][i]['biases'],
                name='<b>{}</b> bias'.format(archive), # Style name/legend entry with html tags
                connectgaps=True,# override default to connect the gaps
                mode='lines+markers'
            ))

        fig.update_layout(width=1500,
                          xaxis_title='Years',
                          yaxis_title=ylabs[i])
        fig.write_image((os.path.join(output_dir, '{}.png'.format(fig_names[i]))))
        fig.write_html((os.path.join(output_dir, '{}.html'.format(fig_names[i]))))


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
    fig.write_html((os.path.join(output_dir, '{}.html'.format(fig_name))))
