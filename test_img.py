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

df1 = pd.read_csv('data_test.csv')

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.LUX])
app.layout = html.Div([
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
])
    
def plot_wordcloud(df, dataset = 'Mdtb', label = 'B1L'):
    d = df.conditions[(df.dataset == 'Mdtb')& (df.label == label)]
    wc = WordCloud (
                    background_color = 'white',
                    width = 512,
                    height = 384
                        ).generate(' '.join(d))
    return wc.to_image()

@app.callback(dd.Output('image_wc', 'src'), [dd.Input('image_wc', 'id')])
def make_image(b):
    img = BytesIO()
    plot_wordcloud(df=df1).save(img, format='PNG')
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())
   
if __name__ == "__main__":
    app.run_server(debug=False)