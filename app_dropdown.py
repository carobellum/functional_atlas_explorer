# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import sys
sys.path.append("..")
import pathlib
import pickle
import numpy as np
from util import *
import parcel_hierarchy as ph

# Import Dash dependencies
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc 


with open('data.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    Prop, atlas, cmap, labels, info, profiles, conditions = pickle.load(f)

# for each parcel, get the highest scoring task
datasets = info.datasets.strip("'[").strip("]'").split("' '")

# Collect parcel profile for each task
label_profile = {}
n_highest = 3
for l, label in enumerate(labels):
    if l != 0:
        parcel_no = labels.tolist().index(label) - 1
        profile = ph.show_parcel_profile(
            parcel_no, profiles, conditions, datasets, show_ds='all', ncond=1, print=False)
        highest_conditions = ['{}:{}'.format(datasets[p][:2], ' & '.join(
            prof[:n_highest])) for p, prof in enumerate(profile)]
        label_profile[label] = highest_conditions

labels_alpha = sorted(label_profile.keys())

datasets = info.datasets.strip("'[").strip("]'").split("' '")


app = Dash(__name__,external_stylesheets=[dbc.themes.CYBORG])

region_labels = dcc.Markdown(children=[], id='chosen_region')
dataset = dcc.Markdown(children=[], id='chosen_dataset')



app.layout = html.Div([
    html.Div(children=[
        html.Label('Region'),
        dcc.Dropdown(labels_alpha, id='chosen_region',value='A1L',clearable=False),
    ], style={'padding': 10, 'flex': 1}),

     html.Div(children=[
        html.Label('Dataset'),
        dcc.Dropdown(datasets, id='chosen_dataset'),
    ], style={'padding': 10, 'flex': 1}),


    html.Div(id='region-conditions'),

    
], style={'display': 'flex', 'flex-direction': 'row'})


@app.callback(
    Output(component_id='region-conditions', component_property='children'),
    Input(component_id='chosen_region', component_property='value'),
    Input(component_id='chosen_dataset', component_property='value'))



def print_conditions(input_value,input_value2):
    conditions = label_profile[input_value]
    if input_value2 == 'Mdtb':
        return f'Conditions: {conditions[0]}'
    if input_value2 == 'Pontine':
        return f'Conditions: {conditions[1]}'
    if input_value2 == 'Nishimoto':
        return f'Conditions: {conditions[2]}'
    if input_value2 == 'Ibc':
        return f'Conditions: {conditions[3]}'



if __name__ == '__main__':
    app.run_server(debug=True)
