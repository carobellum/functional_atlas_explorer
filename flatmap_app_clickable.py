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

parcel = Prop.argmax(axis=0)+1


cerebellum = plot_data_flat(parcel,atlas,cmap = cmap,
                    dtype='label',
                    labels=labels,
                    render='plotly')



#start of app
app = Dash(__name__)

region_labels = dcc.Markdown(id='chosen_region')
dataset = dcc.Markdown(id='chosen_dataset')
click_region_labels = dcc.Markdown(id='clicked-region')



app.layout = html.Div([ html.Div([
    html.H1('Functional Atlas Explorer'),

    html.Div([

        dcc.Graph(id="figure-cerebellum", figure=cerebellum,
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

        html.Table([
        html.Tr([html.Td(['1', html.Sup('st')]), html.Td(id='condition-1')]),
            html.Tr([html.Td(['2', html.Sup('nd')]), html.Td(id='condition-2')]),
            html.Tr([html.Td(['3', html.Sup('rd')]), html.Td(id='condition-3')]),
        ], style={'font-size': '32px', "margin-top": "50px"}),

        html.Div([
            html.H4(id='clicked-region'),
        ]),

    ], style={'width': '49%', 'display': 'inline-block'}),

], style={'display': 'flex', 'flex-direction': 'row'})


@app.callback(
    Output(component_id='condition-1', component_property='children'),
    Output(component_id='condition-2', component_property='children'),
    Output(component_id='condition-3', component_property='children'),
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


@app.callback(
    Output('clicked-region', 'children'),
    Input('figure-cerebellum', 'clickData'))
def display_click_data(clickData):
    selected_region = clickData['points'][0]['text']

    return selected_region

if __name__ == '__main__':
    app.run_server(debug=True)
