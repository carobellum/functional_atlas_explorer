# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import sys
sys.path.append("..")
import pathlib
import pickle
import numpy as np
from util import *
import parcel_hierarchy as ph
import base64


# Import Dash dependencies
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from io import BytesIO
from wordcloud import WordCloud


with open('data.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    Prop, atlas, cmap, labels, info, profiles, conditions = pickle.load(f)
# for each parcel, get the highest scoring task
datasets = info.datasets.strip("'[").strip("]'").split("' '")
# Collect parcel profile for each task
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

parcel = Prop.argmax(axis=0)+1


cerebellum = plot_data_flat(parcel,atlas,cmap = cmap,
                    dtype='label',
                    labels=labels,
                    render='plotly')

#start of app
app = Dash(__name__,external_stylesheets=[dbc.themes.LUX])

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

        html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            # html.H5('Word Cloud',className='text-center'),
                            html.Img(id="image_wc"),
                        ])
                    ])
                ],width={'size':12,"offset":0,'order':1},style={'padding-left' : 25,'padding-right' : 25},className='text-center'),
            ])
        ]),

        html.Div([
            html.H4(id='clicked-region'),
        ]),

    ], style={'width': '49%', 'display': 'inline-block'}),

], style={'display': 'flex', 'flex-direction': 'row'})

def plot_wordcloud(df, dset, region):
    # get the region name
    reg = region['points'][0]['text']
    d = df.conditions[(df.dataset == dset)& (df.label == reg)]
    wc = WordCloud (
                    background_color = 'white',
                    width = 512,
                    height = 384
                        ).generate(' '.join(d))
    return wc.to_image()

@app.callback(

    Output('image_wc', 'src'), 
    # Input(component_id='figure-cerebellum', component_property='clickData'),
    Input(component_id='image_wc', component_property='src'),
    Input(component_id='chosen_dataset', component_property='value'),
    Input(component_id='figure-cerebellum', component_property='clickData'))

def make_image(b, dset, region):
    img = BytesIO()
    plot_wordcloud(df, dset, region).save(img, format='PNG')
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())



if __name__ == '__main__':
    app.run_server(debug=True)