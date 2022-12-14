import sys
sys.path.append("..")
import pathlib
import pickle
import numpy as np
from util import *
import os
import parcel_hierarchy as ph
from wordcloud import WordCloud
import parcel_hierarchy as ph
import base64
import dash
import pandas as pd
import numpy as np
import plotly.express as px
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from io import BytesIO
from wordcloud import WordCloud
import base64
import dash.dependencies as dd

# Import Dash dependencies
from dash import Dash, html, dcc
from dash.dependencies import Input, Output

# import nbformat

with open('data.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    Prop, atlas, cmap, labels, info, profiles, conditions = pickle.load(f)

datasets = info.datasets.strip("'[").strip("]'").split("' '")


# Collect parcel profile for each task
label_profile = {}
n_highest = 3
df_dict = {}
df_dict['dataset'] = []
df_dict['label'] = []
df_dict['L'] = []
df_dict['conditions'] = []
df_dict['parcel_no'] = []

for l,label in enumerate(labels):
    if l != 0:

        parcel_no = labels.tolist().index(label)-1
        profile = ph.show_parcel_profile(parcel_no, profiles, conditions, datasets, show_ds='all', ncond=1, print=False)
        highest_conditions = ['{}:{}'.format(datasets[p][:2], ' & '.join(
            prof[:n_highest])) for p, prof in enumerate(profile)]
        label_profile[label]=highest_conditions

        for p in range(len(profile)):
            current_data = profile[p]
            for c in current_data:
                df_dict['dataset'].append(datasets[p])
                df_dict['label'].append(label)
                df_dict['L'].append(l)
                df_dict['conditions'].append(c)
                df_dict['parcel_no'].append(parcel_no)

labels_alpha = sorted(label_profile.keys())
df = pd.DataFrame(df_dict)
# df.to_csv('data_test.csv')
# labels_alpha = sorted(label_profile.keys())

# Plot the parcellation
# Prop = np.array(model.marginal_prob())
parcel = Prop.argmax(axis=0)+1

cerebellum = plot_data_flat(parcel,atlas,cmap = cmap,
                    dtype='label',
                    labels=labels,
                    render='plotly')

# x2011 = df.conditions[(df.dataset == 'Mdtb')& (df.label == 'B1L')]




#start of app
app = dash.Dash(__name__,external_stylesheets=[dbc.themes.LUX])

region_labels = dcc.Markdown(id='chosen_region')
dataset = dcc.Markdown(id='chosen_dataset')



app.layout = html.Div([ html.Div([
    html.H1('Functional Atlas Explorer'),

    html.Div([

        dcc.Graph(id="graph-basic-2", figure=cerebellum,
                clear_on_unhover=False),

                dcc.Tooltip(id="graph-tooltip")])
    ], style={'width': '49%', 'display': 'inline-block'}),
        
    html.Div([

        html.P('Display functions for a selected region and dataset.'),

        html.Div(
        children=[
            html.Label('Region'),
            dcc.Dropdown(labels_alpha, id='chosen_region',
                        value=labels_alpha[0], clearable=False),
        ], style={'padding': 10, 'flex': 1}),

        html.Div(children=[
            html.Label('Dataset'),
            dcc.Dropdown(datasets, id='chosen_dataset',
                        value=datasets[0], clearable=False),
        ], style={'padding': 10, 'flex': 1}),

        html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5('Word Cloud',className='text-center'),
                            html.Img(id="image_wc"),
                        ])
                    ])
                ],width={'size':12,"offset":0,'order':1},style={'padding-left' : 25,'padding-right' : 25},className='text-center'),
            ])
        ]),
        html.Table([
        html.Tr([html.Td(['1', html.Sup('st')]), html.Td(id='condition-1')]),
            html.Tr([html.Td(['2', html.Sup('nd')]), html.Td(id='condition-2')]),
            html.Tr([html.Td(['3', html.Sup('rd')]), html.Td(id='condition-3')]),
        ], style={'font-size': '32px', "margin-top": "50px"})

    ], style={'width': '49%', 'display': 'inline-block'}),

], style={'display': 'flex', 'flex-direction': 'row'})


@app.callback(
    Output(component_id='condition-1', component_property='children'),
    Output(component_id='condition-2', component_property='children'),
    Output(component_id='condition-3', component_property='children'),
    # Output(component_id = "word cloud", component_property = 'Img'),
    Input(component_id='chosen_region', component_property='value'),
    Input(component_id='chosen_dataset', component_property='value'))


def print_conditions(region,dset):
    conditions = label_profile[region]
    # Find which condition list is the one of the chosen dataset
    dset_short = dset[:2]
    match = [idx for idx, list in enumerate(conditions) if dset_short in list]
    conditions_dset = conditions[match[0]]
    # Now format conditions for printing
    conditions_dset = conditions_dset[3:]
    conditions_dset = conditions_dset.split('&')

    return conditions_dset[0], conditions_dset[1], conditions_dset[2]

def plot_wordcloud(df, dset, region):
    d = df.conditions[(df.dataset == dset)& (df.label == region)]
    wc = WordCloud (
                    background_color = 'white',
                    width = 512,
                    height = 384
                        ).generate(' '.join(d))
    return wc.to_image()

@app.callback(
    Output('image_wc', 'src'), 
    # Input('image_wc', 'chosen_dataset', 'chosen_region'),
    Input(component_id='image_wc', component_property='src'),
    Input(component_id='chosen_dataset', component_property='value'),
    Input(component_id='chosen_region', component_property='value'),
    )
def make_image(b, dset, region):
    img = BytesIO()
    plot_wordcloud(df, dset, region).save(img, format='PNG')
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())


if __name__ == '__main__':
    app.run_server(debug=True)
