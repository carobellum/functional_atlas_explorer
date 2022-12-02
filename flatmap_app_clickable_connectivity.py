# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import sys
sys.path.append("..")
import pathlib
import pickle
import numpy as np
from util import *
import parcel_hierarchy as ph
import dash_bootstrap_components as dbc


# Import Dash dependencies
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
from PIL import Image


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

# Define a dictionary for mapping the regions to connectivity profiles
# maps = np.()

map_files = np.array(['connectivity_images/Action_Observation.png', 'connectivity_images/Active_Maintenance.png', 'connectivity_images/Autobiographic_Recall.png', 'connectivity_images/Divided_Attention.png', 'connectivity_images/Left_Hand.png', 'connectivity_images/Narrative.png', 'connectivity_images/Right_Hand.png', 'connectivity_images/Saccades.png', 'connectivity_images/Semantic_Knowledge.png', 'connectivity_images/Verbal_Fluency.png', \
                           'connectivity_images/Action_Observation.png', 'connectivity_images/Active_Maintenance.png', 'connectivity_images/Autobiographic_Recall.png', 'connectivity_images/Divided_Attention.png', 'connectivity_images/Left_Hand.png', 'connectivity_images/Narrative.png', 'connectivity_images/Right_Hand.png', 'connectivity_images/Saccades.png', 'connectivity_images/Semantic_Knowledge.png', 'connectivity_images/Verbal_Fluency.png', \
                           'connectivity_images/Action_Observation.png', 'connectivity_images/Active_Maintenance.png', 'connectivity_images/Autobiographic_Recall.png', 'connectivity_images/Divided_Attention.png', 'connectivity_images/Left_Hand.png', 'connectivity_images/Narrative.png', 'connectivity_images/Right_Hand.png', 'connectivity_images/Saccades.png', 'connectivity_images/Semantic_Knowledge.png', 'connectivity_images/Verbal_Fluency.png', \
                           'connectivity_images/Action_Observation.png', 'connectivity_images/Active_Maintenance.png', 'connectivity_images/Autobiographic_Recall.png', 'connectivity_images/Divided_Attention.png'])
connectivity = dict(map(lambda i, j: (i, j), labels_alpha, map_files.tolist()))



#start of app
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY]
           )
# region_labels = dcc.Markdown(id='chosen_region')
# dataset = dcc.Markdown(id='chosen_dataset')
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

        # for Dash version < 2.2.0
        html.Img(id='chosen-connectivity'),


    ], style={'width': '49%', 'display': 'inline-block'}),

], style={'display': 'flex', 'flex-direction': 'row'})

# Condition Map Updating on Click
@app.callback(
    Output(component_id='chosen-connectivity', component_property='src'),
    Input(component_id='figure-cerebellum', component_property='clickData'))
def show_connectivity(region):
    # When initiliazing the website and if clickin on a null region, show default image
    connectivity_image = Image.open(
        'connectivity_images/Action_Observation.png')
    if region is not None and region['points'][0]['text'] != '0':
        label = region['points'][0]['text']
        conn_region = connectivity[label]
        connectivity_image = Image.open(conn_region)

    return connectivity_image


# Condition Printing on Click
@app.callback(
    Output(component_id='condition-1', component_property='children'),
    Output(component_id='condition-2', component_property='children'),
    Output(component_id='condition-3', component_property='children'),
    Input(component_id='figure-cerebellum', component_property='clickData'),
    Input(component_id='chosen_dataset', component_property='value'))

def print_conditions(region,dset):
    conditions_dset = ['', '', ''] # When initiliazing the website and if clickin on a null region, show no conditions
    if region is not None and region['points'][0]['text'] != '0':
        label = region['points'][0]['text']
        conditions = label_profile[label]
        # Find which condition list is the one of the chosen dataset
        dset_short = dset[:2]
        match = [idx for idx, list in enumerate(conditions) if dset_short in list]
        conditions_dset = conditions[match[0]]
        # Now format conditions for printing
        conditions_dset = conditions_dset[3:]
        conditions_dset = conditions_dset.split('&')

    return conditions_dset[0], conditions_dset[1], conditions_dset[2]



if __name__ == '__main__':
    app.run_server(debug=True)
