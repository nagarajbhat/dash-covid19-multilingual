# -*- coding: utf-8 -*-

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import time
#import torch
from transformers import MarianMTModel, MarianTokenizer
#import nltk
from typing import List
import numpy as np
#import json
import plotly
from newsapi import NewsApiClient
import datetime
import urllib.request as requests



#-------------------------------------------------------------------------------------------------------
#functions
def get_model(src_lang,trg_lang):
    src = src_lang
    trg = trg_lang
    #perform a check to see if target language is english
    if trg_lang == 'en':
        model,tok= 0,0
    else:
        mname = f'Helsinki-NLP/opus-mt-{src}-{trg}'
        model = MarianMTModel.from_pretrained(mname)
        tok = MarianTokenizer.from_pretrained(mname)
    return model,tok

def translate(model,tok,src_text):
    #if model=0 (i.e target language was selected as english , return words as src txt)
    if model==0:
        lst = ['']
        lst[0] = src_text #we store src text in a list[0] since we use words[0] everywhere
        words = lst
    else:
        batch = tok.prepare_translation_batch(src_texts = [src_text])
        gen = model.generate(**batch)
        words: List[str] = tok.batch_decode(gen,  skip_special_tokens=True)
    return words

#model,tok = get_model('en','de')

"""
#English to other language supported
lang_supported = [
    'en','af','bcl','bem','ber','bg','br',
    'af','bi','bzs','ca','ceb','chk','crs','cs',
    'cy', 'da', 'de', 'ee','efi','el'
    'eo','et','eu','fi','fj','fr','ga',
    'gaa','gil','gl','guw','gv','ha','he',
    'hil','ho','ht','hu','id','ig','ilo',
    'is','iso','it','jap','kg','kj','kqn',
    'kwn','kwy','lg','ln','loz','lu','lua','lue',
    'lun','luo','lus','mfe','mg','mh','mk','ml',
    'mos','mr','mt','ng','niu','nl','nso','ny',
    'nyk','om','pag','pap','pis','pon','rnd','ro','ru','run',
    'rw','sg','sk','sm','sn','sq','ss',
    'st','sv','sw','swc','tdt','ti','tiv',
    'tl','tll','tn','to','toi','tpi','ts','tvl',
    'tw','ty','uk','umb','xh' 
]
"""
#Approved languages - languages that give good results
# {'de':'German','fr':'French'}

#bad results
# {'mr':'Marathi','jap':'Japanese'}

#langs = ['en','fr','de']

#langs_dict = {'en':'English','de':'German','fr':'French','ru':'Russian','ga':'Irish','da':'Danish','id':'Indonesian'}
#langs_dict = {'en':'English','de':'German','fr':'French'}

langs_dict = {'en':'English'}

#-------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------
# df - live integration
url = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'
response = requests.urlopen(url)
df_live = pd.read_csv(response)
df= df_live
#df = pd.read_csv('./data/owid-covid-data.csv')

#create more dataframes
df_na = df.dropna(subset=['total_cases','new_cases','total_deaths','new_deaths'])
df_na['date'] = pd.to_datetime(df_na['date'])
data_latest = df_na[df_na['date']==df_na['date'].drop_duplicates().nlargest(1).iloc[-1]]

# dataframe for map
country_codes = pd.read_csv('./data/countries_codes.csv')
df_code  = country_codes.filter(['alpha2','alpha3'])
df_code['alpha3'] = df_code['alpha3'].str.upper()
df_code['alpha2'] = df_code['alpha2'].str.upper()

df_merged = df.merge(df_code,left_on='iso_code',right_on='alpha3') #merging country codes

df_loc = pd.read_csv('./data/countries.csv')
df_loc = df_loc.filter(['country','latitude','longitude']) 
df_map = df_merged.merge(df_loc,left_on='alpha2',right_on='country') #merging lat long

df_map_latest = df_map[df_map['date']==df_map['date'].max()]



#variables
graph_template = "plotly_dark"
ext_stylesheet = dbc.themes.DARKLY

criteria = {'total_cases':'Total cases','total_deaths':'Total Deaths','total_tests':'Total tests'}
continents = ['All','Asia','Europe','Africa','North America','South America','Oceania']

cols = plotly.colors.DEFAULT_PLOTLY_COLORS


## variables - news api
newsapi = NewsApiClient(api_key='020d9a51f8d5433cb6a571b7ca777088')

news_language_list = {'ar':'Arabic','de':'German','en':'English','es':'Spanish','fr':'French','he':'Hebrew',
                'it':'Italian','nl':'Dutch','no':'Norwegian','pt':'Portuguese','ru':'Russian','se':'Northern Sami',
                'ud':'Udmurt','zh':'Chinese'}

"""
all_articles = newsapi.get_everything(q='bitcoin',
                                      sources='bbc-news,the-verge',
                                      domains='bbc.co.uk,techcrunch.com',
                                      from_param='2017-12-01',
                                      to='2017-12-12',
                                      language='en',
                                      sort_by='relevancy',
                                      page=2)
# /v2/sources
sources = newsapi.get_sources()
"""


"""
pre-train
We are pretraining model for each translations to reduce loading time (as it takes a lot of time)
format is pretrain[<language code>][model_tok][0] to access model
and pretrain[<language code>][model_tok][1] to access tok( token)
this way we don't even have to connect two callbacks
#this means the program execution time will be high, but app will load quickly
"""
pretrain = {i:{'model_tok':get_model('en',i)} for i in langs_dict.keys()}

#pretranslated = {i:{'translated_text':translate(pretrain[i]['model_tok'][0],
 #                                               pretrain[i]['model_tok'][1]) for i in lang_dict.keys()}}
 
#------------------------------------------------------------------------------------------------------------
#app
app = dash.Dash('dash-covid19-translator',
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
                external_stylesheets=[ext_stylesheet])
app.title = "Multilingual Covid-19 Dashboard"

app_name = "Multilingual Covid-19 Dashboard" 

server = app.server

#---------------------------------------------------------------------------------------------------
#controls
controls_1a = dbc.Form(
        [
                dbc.FormGroup([
                    dbc.Label(id='label_select_language',children=['Select language']),
                    dcc.Dropdown(
                            id = 'dropdown_language',
                            options = [{'label':langs_dict[i],'value':i} for i in langs_dict.keys()],
                            value = list(langs_dict.keys())[0],
                             style={'color':'black'}
                            )
                    ]
        ),
                dbc.Spinner(color="secondary",type="grow",children=[dbc.Card(dbc.CardBody(id='language_updated'))   ])

             ]),

               

controls_1b =    dbc.Form([
                 dbc.FormGroup([
            #dbc.Label(translate(model,tok,'Select continent')[0]),
            dbc.Label(id='label_select_continent',children=["select the continent"]),
            dcc.Dropdown(
                    id='dropdown_continent',
                    options=[{'label':i,'value':i} for i in continents],
                    value='All',
                    style={'color':'black'}
)
            ]
        ),

                dbc.FormGroup([
            #dbc.Label(translate(model,tok,'Select continent')[0]),
            dbc.Label(id='label_select_country',children=["select the country"]),
            dcc.Dropdown(
                    id='dropdown_country',
                    options=[{'label':i,'value':i} for i in df['location'].unique()],
                    value='World',
                    style={'color':'black'}

                     
                     )
            ]
        )
                ])


controls_1c =     dbc.Form([
                    dbc.FormGroup([ 
            #dbc.Label(translate(model,tok,'Select continent')[0]),
            dbc.Label(id='label_select_criteria',children=["select the criteria"]),
            dcc.Dropdown(
                    id='dropdown_criteria',
                    options=[{'label':i,'value':i} for i in criteria.keys()],
                    value='total_cases',
                    style={'color':'black'}
)
            ]
        )
              
      ])

controls_2a = dbc.Form(
        [
                dbc.FormGroup([
                    dbc.Label(id='label_news_language',children=['Select news language']),
                    dcc.Dropdown(
                            id = 'dropdown_news_language',
                            options = [{'label':news_language_list[i],'value':i} for i in news_language_list.keys()],
                            value = 'en',
                             style={'color':'black'}
                            )
                    ]
        ),
                dbc.Spinner(color="secondary",type="grow",children=[dbc.Card(dbc.CardBody(id='news_language_updated'))   ])

             ]),



#tabs
tab1 = dbc.Card([
        dbc.CardBody([
                #controls
                dbc.Row([
                dbc.Col(controls_1a,md=4),
                dbc.Col(controls_1b,md=4),
                dbc.Col(controls_1c,md=4),
    
                ]),

                #info cards
                dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader(html.H6(id='label_info1',children=['location country'])),
                    dbc.CardBody(html.H4(id='location_country'))
                    ],
                    color='info',
                    )),
                 dbc.Col(dbc.Card([
                    dbc.CardHeader(html.H6(id='label_info2',children=['date'])),
                    dbc.CardBody(html.H4(id='date'))
                    ],
                    color='info',
                    )),
                  dbc.Col(dbc.Card([
                    dbc.CardHeader(html.H6(id='label_info3',children=['Total cases'])),
                    dbc.CardBody(html.H4(id='total_cases'))
                    ],
                    color='warning',
                    )),
                   dbc.Col(dbc.Card([
                    dbc.CardHeader(html.H6(id='label_info4',children=['New cases'])),
                    dbc.CardBody(html.H4(id='new_cases'))
                    ],
                    color='warning',
                    )),
                    dbc.Col(dbc.Card([
                    dbc.CardHeader(html.H6(id='label_info5',children=['total deaths'])),
                    dbc.CardBody(html.H4(id='total_deaths'))
                    ],
                    color='danger',
                    )),
                     dbc.Col(dbc.Card([
                    dbc.CardHeader(html.H6(id='label_info6',children=['New deaths'])),
                    dbc.CardBody(html.H4(id='new_deaths'))
                    ],
                    color='danger',
                    )),
                      dbc.Col(dbc.Card([
                    dbc.CardHeader(html.H6(id='label_info7',children=['total tests'])),
                    dbc.CardBody(html.H4(id='total_tests'))
                    ],
                    color='success',
                    )),
                       dbc.Col(dbc.Card([
                    dbc.CardHeader(html.H6(id='label_info8',children=['New tests'])),
                    dbc.CardBody(html.H4(id='new_tests'))
                    ],
                    color='success',
                    )),
               
                ]),
                html.Br(),

                #graphs
                dbc.Spinner(color="primary",type="grow",children=[dcc.Graph(id='graph_map_continent')]),
                html.Br(),
                dbc.Spinner(color="primary",type="grow",children=[dcc.Graph(id='graph_treemap_continent')]),
                html.Br(),
                dbc.Spinner(color="primary",type="grow",children=[dcc.Graph(id='graph_bar_continent')]),
                html.Br(),
                
                dbc.Spinner(color="primary",type="grow",children=[dcc.Graph(id='graph_line_continent')]),
                html.Br(),
                dbc.Spinner(color="primary",type="grow",children=[dcc.Graph(id='graph_area_continent')]),
                html.Br(),
                dbc.Spinner(color="primary",type="grow",children=[dcc.Graph(id='graph_compare_country')]),
                html.Br(),
               
                html.H6(id="label_not_translated",children=["If any text in this dashboard is untranslated, type or copy paste it here this to translate!"]),
                dcc.Input(id="input_text", type="text", placeholder=""),
                html.Button('Translate', id='submit-val'),
                dbc.Spinner(color="primary",type="grow",children=[html.Div(id='output_text')])

                

                
                ])
        ])

tab2 = dbc.Card([
        dbc.CardBody([
        
            # Init
        dbc.Row([
                dbc.Col(controls_2a,md=4)
                
    
                ]),
        dbc.Spinner(color="primary",type="grow",children=html.Div(id="news_all")),
        #html.Div(all_articles)

       
            ])
    ])

#nav
PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"


navbar = dbc.Navbar(
    [
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                   # dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                    dbc.Col(dbc.NavbarBrand("Multilingual Covid-19 Dashboard", className="ml-2",style={"font-size":25})),
                ],
                align="center",
                no_gutters=True,
            ),
            href="https://plot.ly",
        ),
        dbc.NavbarToggler(id="navbar-toggler"),
    ],
    color="dark",
    dark=True,
    style={'width':'100%'}
)

#app layout
app.layout = dbc.Container([
            dbc.Row(navbar),

          dbc.Tabs([
                 dbc.Tab(tab1, id="label_tab1",label='Analysis'),
                dbc.Tab(tab2, id="label_tab2",label="Covid headlines"),

                 
        ])
        ],
            fluid=True
)

    



#-------------------------------------------------------------------------------------------
#callbacks
@app.callback([Output('language_updated','children')],
              [Input('dropdown_language','value')])
def set_language(trg_language):
    #global model, tok
    #model,tok = get_model('en',trg_language)
    model = pretrain[trg_language]['model_tok'][0]
    tok = pretrain[trg_language]['model_tok'][1]
    translate(model,tok,'Total cases of Covid 19 by continent')
    trg_language_label = langs_dict[trg_language]
    language_updated = [f'language has been updated to {trg_language_label}']
    return language_updated

@app.callback([Output('label_select_language','children'),
               Output('label_select_continent','children'),
             Output('label_select_country','children'),
               Output('label_select_criteria','children'),

                Output('label_info1','children'),
                Output('label_info2','children'),
                Output('label_info3','children'),
                Output('label_info4','children'),
                Output('label_info5','children'),
                Output('label_info6','children'),
                Output('label_info7','children'),
                Output('label_info8','children'),
                 Output('label_not_translated','children'),


               ],
              [Input('dropdown_language','value')])
def set_lables(trg_language):
    #model = pretrain[trg_language]['model_tok'][0]
    #tok = pretrain[trg_language]['model_tok'][1]    
    #label_tab1 = translate(model,tok,'Continent analysis')
    #label_select_language = translate(model,tok,'Select language')
    #label_select_continent = translate(model,tok,'Select the continent')
    #label_select_country = translate(model,tok,'Select the country')
    #label_select_criteria = translate(model,tok,'Select the criteria')

    #label_info1 = translate(model,tok,"Location/Country")
    #label_info2 = translate(model,tok,"Date")
    #label_info3 = translate(model,tok,"total_cases")
    #label_info4 = translate(model,tok,"New cases")
    #label_info5 = translate(model,tok,"Total deaths")
    #label_info6 = translate(model,tok,"New deaths")
    #label_info7 = translate(model,tok,"total tests")
    #label_info8 = translate(model,tok,"New tests")
    #label_not_translated = translate(model,tok,'If any text in this dashboard is untranslated, type or copy paste it here this to translate.')

    label_select_language = 'Select language'
    label_select_continent = 'Select the continent'
    label_select_country = 'Select the country'
    label_select_criteria = 'Select the criteria'

    label_info1 = "Location/Country"
    label_info2 = "Date"
    label_info3 = "total_cases"
    label_info4 = "New cases"
    label_info5 = "Total deaths"
    label_info6 = "New deaths"
    label_info7 = "total tests"
    label_info8 = "New tests"
    label_not_translated = 'If any text in this dashboard is untranslated, type or copy paste it here this to translate.'

    return label_select_language,label_select_continent,label_select_country,label_select_criteria,label_info1,label_info2,label_info3,label_info4,label_info5,label_info6,label_info7,label_info8,label_not_translated
    
@app.callback([Output('dropdown_continent','value'),
                Output('location_country','children'),
                Output('date','children'),
               Output('total_cases','children'),
               Output('new_cases','children'),
               Output('total_deaths','children'),
               Output('new_deaths','children'),
               Output('total_tests','children'),
               Output('new_tests','children')],
              [Input('dropdown_country','value')])
def info_cards(location_country):
    data_world = data_latest[data_latest['location']==location_country]
    location_continent='Asia'
    if location_country!='World':
        location_continent = data_world['continent'].values[0]
    else:
        location_continent = 'All'
    #get_latest_date
    date = data_world['date'].dt.date
    total_cases = data_world['total_cases']
    new_cases = data_world['new_cases']
    total_deaths = data_world['total_deaths']
    new_deaths = data_world['new_deaths']
    total_tests = data_world['total_tests']
    new_tests = data_world['new_tests']
    print(type(total_tests.values[0]))
    if np.isnan(total_tests.values[0]):
        total_tests = 'Not available'
    if np.isnan(new_tests.values[0]):
        new_tests = 'Not available'

    return location_continent,location_country,date, total_cases, new_cases, total_deaths, new_deaths, total_tests, new_tests


    
    
    
    
    
    
@app.callback([Output('graph_line_continent','figure'),
                Output('graph_area_continent','figure')],
              [Input('dropdown_continent','value'),
               Input('dropdown_language','value'),
               Input('dropdown_criteria','value'),
               ])
def line_graphs(continent,trg_language,criteria):
    #line graph - continent
    if continent !='All':
        data = df[df['continent']==continent]
    else:
        data = df
    #data = data.groupby(['location','date']).sum().reset_index()
    data = data.dropna(subset=["location","continent"])

    data = data[data.location != 'World']
    line_fig = px.line(data,x='date',y=criteria,color='location',template = graph_template)
    model = pretrain[trg_language]['model_tok'][0]
    tok = pretrain[trg_language]['model_tok'][1]
    
    text_title = translate(model,tok,f'Time line of {criteria} of Covid 19 by continent - {continent}')
    text_xaxis = translate(model,tok,'date')
    text_yaxis = translate(model,tok,criteria)
    text_legend = translate(model,tok,'location')
    line_fig.update_traces(textfont_size=30)

    line_fig.update_layout(title=text_title[0], xaxis_title=text_xaxis[0],
    yaxis_title=text_yaxis[0],legend_title=text_legend[0])

   #Area
    area_fig = px.area(data,x="date", y=criteria, color="continent",line_group="location",template = graph_template)
    text_title = translate(model,tok,f'Area graph - Time line of {criteria} of Covid 19 by continent - {continent}')
    text_xaxis = translate(model,tok,'date')
    text_yaxis = translate(model,tok,criteria)
    text_legend = translate(model,tok,'location')
    area_fig.update_layout(title=text_title[0], xaxis_title=text_xaxis[0],
    yaxis_title=text_yaxis[0],legend_title=text_legend[0])

    return line_fig,area_fig

@app.callback(Output('graph_compare_country','figure'),
              [Input('dropdown_continent','value'),
               Input('dropdown_language','value'),
               Input('dropdown_criteria','value'),
               ])
def country_graphs(continent,trg_language,criteria):
    #line graph - continent
    if continent !='All':
        data = df[df['continent']==continent]
    else:
        data = df
    #data = data.groupby(['location','date']).sum().reset_index()
   # data = data.dropna(subset=["location","continent"])
    data = data.groupby(['continent','date']).sum().reset_index()
    #data = data[data['location']==country]

    #new cases
    area_fig = go.Figure()
    area_fig.add_trace(go.Scatter(x=data['date'],y=data['total_cases'],name='total cases',mode='lines',fill='tonexty',line=dict(width=2, color=cols[0])))
    area_fig.add_trace(go.Scatter(x=data['date'],y=data['total_deaths'],name='total deaths',mode='lines',fill='tozeroy',line=dict(width=2, color=cols[1])))
    
    model = pretrain[trg_language]['model_tok'][0]
    tok = pretrain[trg_language]['model_tok'][1]
    
    text_title = translate(model,tok,f'Time line of Cases vs Deaths of Covid 19 by continent - {continent}')
    text_xaxis = translate(model,tok,'date')
    text_yaxis = translate(model,tok,'people')
    area_fig.update_layout(template = graph_template,title=text_title[0], xaxis_title=text_xaxis[0],
    yaxis_title=text_yaxis[0])
    return area_fig



@app.callback([Output('graph_bar_continent','figure'),
              Output('graph_treemap_continent','figure')],
              [Input('dropdown_continent','value'),
               Input('dropdown_language','value'),
               Input('dropdown_criteria','value'),
               ])
def bar_graphs(continent,trg_language,criteria):
    model = pretrain[trg_language]['model_tok'][0]
    tok = pretrain[trg_language]['model_tok'][1]
    #bar graph - continent 
    if continent !='All':
        data = data_latest[data_latest['continent']==continent]
    else:
        data = data_latest
    data = data[data.location != 'World']
    bar_fig = px.bar(data, x="location", y=criteria,color="population",color_continuous_scale=px.colors.sequential.YlOrRd)
    text_title = translate(model,tok,f'{criteria} vs population of Covid 19 - {continent}')
    text_xaxis = translate(model,tok,'countries')
    text_yaxis = translate(model,tok,criteria)
    text_legend = translate(model,tok,'continent')
    bar_fig.update_layout(title=text_title[0], xaxis_title=text_xaxis[0],
    yaxis_title=text_yaxis[0],legend_title=text_legend[0],height=500,template = graph_template)

    #Treemap
    #data = data.dropna(subset=['total_cases','total_deaths'])
    if continent !='All':
        data = data_latest[data_latest['continent']==continent]
    else:
        data = data_latest  
    data = data[data.location != 'World']

    treemap_fig = px.treemap(data, path=['continent','location'], values=criteria,color=criteria,height=600,template = graph_template,
                            color_continuous_scale=px.colors.sequential.YlOrRd)
    
    text_title = translate(model,tok,f'Tree map of {criteria} by continent - {continent}')
    text_legend = translate(model,tok,criteria)

    treemap_fig.update_layout(title = text_title[0],legend_title=text_legend[0])
    return bar_fig,treemap_fig



@app.callback(Output('graph_map_continent','figure'),
              [Input('dropdown_continent','value'),
               Input('dropdown_language','value'),
               Input('dropdown_criteria','value'),
               ])
def map_graph(continent,trg_language,criteria):
    #line graph - continent
    if continent !='All':
        data = df_map_latest[df_map_latest['continent']==continent]
    else:
        data = df_map_latest
    #data = data.groupby(['location','date']).sum().reset_index()
    #data = data[data.location != 'World']
    data = data.dropna(subset=['total_deaths','total_cases'])
    fig_map = px.scatter_mapbox(data, lat="latitude", lon="longitude", color=criteria,size=criteria,
                        size_max=40,mapbox_style="carto-darkmatter",
                       # animation_frame = 'date', animation_group = 'location',
                        zoom=1.6,height=800,template = graph_template, color_continuous_scale=px.colors.sequential.YlOrRd,
                       hover_data=['location','total_cases','total_deaths'])
    #fig_map.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 0.0000004
    
    model = pretrain[trg_language]['model_tok'][0]
    tok = pretrain[trg_language]['model_tok'][1]
    
    text_title = translate(model,tok,f'Map of {criteria} of Covid 19 by continent - {continent}')
    text_xaxis = translate(model,tok,'date')
    text_yaxis = translate(model,tok,criteria)
    text_legend = translate(model,tok,'location')
    fig_map.update_layout(title=text_title[0], xaxis_title=text_xaxis[0],
    yaxis_title=text_yaxis[0],legend_title=text_legend[0])
    #mapbox_accesstoken="pk.eyJ1IjoibmFnYXJhamJoYXQiLCJhIjoiY2tjeTJibTEwMDVzOTM1azczdHppeGw4dCJ9.a_o1eljq2K4DL0m9iu5F3A"
    #mapbox_layers=[
     #   {
      #      "below": 'traces',
       #     "sourcetype": "raster",
         #   "source": [
          #      "https://basemap.nationalmap.gov/arcgis/rest/services/DeLorme_World_Base_Map/MapServer/tile/{z}/{y}/{x}"
           # ]
        #}
      #]
      
    # options - USGSImageryOnly
    
    return fig_map



@app.callback(Output('output_text','children'),
              [Input('submit-val','n_clicks'),
               Input('dropdown_language','value')],
              [State('input_text','value')])
def text_translation(n_clicks,trg_language,input_text):
    model = pretrain[trg_language]['model_tok'][0]
    tok = pretrain[trg_language]['model_tok'][1]
    output = translate(model,tok,input_text)
    return output
#there is one way to make it work currently and that is to remove set_language and 
#pass dropdown_language directly to graph_update    
    


@app.callback(Output('news_all','children'),
               [Input('dropdown_news_language','value')])
def news_update(trg_language):
    #model = pretrain[trg_language]['model_tok'][0]
    #tok = pretrain[trg_language]['model_tok'][1]
    
    # /v2/top-headlines
    top_headlines = newsapi.get_top_headlines(q='Covid',
                                          #sources='bbc-news,the-verge',
                                          #category='business',
                                          language=trg_language)
                                          #country='us')
    #mod_tok1 = get_model('en','fr')

    #top_headlines_title = [str(translate(model,tok,top_headlines['articles'][i]['title'])[0]) for i in range(len(top_headlines['articles']))]
    top_headlines_title = [str(top_headlines['articles'][i]['title']) for i in range(len(top_headlines['articles']))]
    
    top_headlines_url = [top_headlines['articles'][i]['url'] for i in range(len(top_headlines['articles']))]
    top_headlines_source = [str(top_headlines['articles'][i]['source']['name']) for i in range(len(top_headlines['articles']))]
    top_headlines_content = [str(top_headlines['articles'][i]['content']).split("[")[0] for i in range(len(top_headlines['articles']))]
    top_headlines_description = [str(top_headlines['articles'][i]['description']) for i in range(len(top_headlines['articles']))]
    top_headlines_date =  [str(datetime.datetime.date(pd.to_datetime(top_headlines['articles'][i]['publishedAt']))) for i in range(len(top_headlines['articles']))]
    top_headlines_img =  [str(top_headlines['articles'][i]['urlToImage']) for i in range(len(top_headlines['articles']))]
    
    news_all= [dbc.Card([
                dbc.CardHeader([html.A(id="news_title_url",children=[top_headlines_title[i]],href=top_headlines_url[i])
                                    ,html.P("|"+top_headlines_source[i]+"|"+top_headlines_date[i])]),
                dbc.CardBody([html.P(id="news_source",children=top_headlines_description[i]),
                              html.Img(src=top_headlines_img[i],alt="image",height="80%", width="80%")]),
                ],
                color='secondary',style={"width":400,"height":400,"display":"flex","float":"left",  "margin": 20}) for i in range(len(top_headlines_title))]
    #top_headlines_sources
    # /v2/everything
    return news_all
#there is one way to make it work currently and that is to remove set_language and 
#pass dropdown_language directly to graph_update    

if __name__=='__main__':
    app.run_server(debug=False)



