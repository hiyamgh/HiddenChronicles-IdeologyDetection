import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd

df = px.data.stocks()#.tail(50)
df = df.drop(['date'], axis = 1)
dfc = df.corr()
z = dfc.values.tolist()

# change each element of z to type string for annotations
# z_text = [[str(y) for y in x] for x in z]
z_text = [[str(round(y, 1)) for y in x] for x in z]
# df.columns =['2014', '2015', '2016', '2017', '2018', '2019']
dfcolumns =[1960, 1962, 1964, 1966, 1968, 1970]
# df.columns =['y1960', 'y1962', 'y1964', 'y1966', 'y1968', 'y1970']

# set up figure
fig = ff.create_annotated_heatmap(z=z_text, x=list(dfcolumns),
                                     y=list(dfcolumns))

# add custom xaxis title

fig.add_shape(type="rect",
              x0=-0.5, y0=1.5, x1=3.5, y1=5.5,
              line=dict(color="blue", width = 4),
              )

fig.add_shape(type="rect",
              x0=3.5, y0=-0.5, x1=5.5, y1=1.5,
              line=dict(color="green", width = 4),
              )


# adjust margins to make room for yaxis title
fig.update_layout(margin=dict(t=50, l=200))
fig.update_shapes(dict(xref='x', yref='y'))

# add colorbar
fig['data'][0]['showscale'] = True
fig.update_xaxes(type='category')
fig.update_yaxes(type='category')
fig.show()