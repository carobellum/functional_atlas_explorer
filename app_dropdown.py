# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import sys
sys.path.append("..")
import pathlib
import pickle
import numpy as np
from util import *
import parcel_hierarchy as ph
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
# import nbformat

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

children = label_profile

app = Dash(__name__)

app.layout = html.Div([
    html.Div(children=[
        html.Label('Region'),
        dcc.Dropdown(labels_alpha),
    ], style={'padding': 10, 'flex': 1}),

    
], style={'display': 'flex', 'flex-direction': 'row'})


@app.callback(
    Output(label_profile[label][0], 'children'),
    Input(labels_alpha, 'selectedData'))
def display_selected_data(labels_alpha):
    return print(labels_alpha, indent=2)

if __name__ == '__main__':
    app.run_server(debug=True)
