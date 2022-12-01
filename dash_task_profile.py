import sys
sys.path.append("..")
import pathlib
import pickle
import numpy as np
from util import *
import parcel_hierarchy as ph
from dash import Dash, html, dcc, Input, Output
import pandas as pd
import plotly.express as px

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

def get_profile():

    with open('data.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        Prop, atlas, cmap, labels, info, profiles, conditions = pickle.load(f)

    parcel = Prop.argmax(axis=0)+1


    ax = plot_data_flat(parcel,atlas,cmap = cmap,
                        dtype='label',
                        labels=labels,
                        render='plotly')
    ax.show()

    # for each parcel, get the highest scoring task
    datasets = info.datasets.strip("'[").strip("]'").split("' '")
    # Collect parcel profile for each task
    label_profile = {}
    n_highest = 3
    for l,label in enumerate(labels):
        if l != 0:
            parcel_no = labels.tolist().index(label)-1
            profile = ph.show_parcel_profile(parcel_no, profiles, conditions, datasets, show_ds='all', ncond=1, print=False)
            highest_conditions = ['{}:{}'.format(datasets[p][:2], ' & '.join(prof[:n_highest])) for p,prof in enumerate(profile)]
            label_profile[label]=highest_conditions

    labels_alpha = sorted(label_profile.keys())

    return labels_alpha

# app.layout = html.Div([
#     html.Div([

#         html.Div([
#             dcc.Dropdown(
#                 df['Indicator Name'].unique(),
#                 'Fertility rate, total (births per woman)',
#                 id='crossfilter-xaxis-column',
#             ),
#             dcc.RadioItems(
#                 ['Linear', 'Log'],
#                 'Linear',
#                 id='crossfilter-xaxis-type',
#                 labelStyle={'display': 'inline-block', 'marginTop': '5px'}
#             )
#         ],
#         style={'width': '49%', 'display': 'inline-block'}),

#         html.Div([
#             dcc.Dropdown(
#                 df['Indicator Name'].unique(),
#                 'Life expectancy at birth, total (years)',
#                 id='crossfilter-yaxis-column'
#             ),
#             dcc.RadioItems(
#                 ['Linear', 'Log'],
#                 'Linear',
#                 id='crossfilter-yaxis-type',
#                 labelStyle={'display': 'inline-block', 'marginTop': '5px'}
#             )
#         ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
#     ], style={
#         'padding': '10px 5px'
#     }),

#     html.Div([
#         dcc.Graph(
#             id='crossfilter-indicator-scatter',
#             hoverData={'points': [{'customdata': 'Japan'}]}
#         )
#     ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
#     html.Div([
#         dcc.Graph(id='x-time-series'),
#         dcc.Graph(id='y-time-series'),
#     ], style={'display': 'inline-block', 'width': '49%'}),

#     html.Div(dcc.Slider(
#         df['Year'].min(),
#         df['Year'].max(),
#         step=None,
#         id='crossfilter-year--slider',
#         value=df['Year'].max(),
#         marks={str(year): str(year) for year in df['Year'].unique()}
#     ), style={'width': '49%', 'padding': '0px 20px 20px 20px'})
# ])
if __name__ == '__main__':
    labels_alpha = get_profile()
    # Plot the parcellation
    # Prop = np.array(model.marginal_prob())
    
    print("hello")