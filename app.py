import pickle
from flask import Flask, render_template, request, jsonify, url_for
from PIL import Image
import os
import numpy as np
import time
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model

# Falcon9 Plotly dash imports
from dash import Dash, html, dcc, callback, Output, Input, ctx, State
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
import csv
import sqlite3
import sqlalchemy
import datetime

# Recomender system imports
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer # Document Term Frequency
import rake_nltk
from rake_nltk import Rake
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# CO2 emission
import folium

# fetch data from SQlite database abd create a dataframe
con = sqlite3.connect("./dashboards/falcon9/falcon9.db")
cur = con.cursor()

sql_query = "SELECT * FROM `falcon9_tbl`"

df = pd.read_sql(sql=sql_query, con=con)
df['Flight_No'] = range(1, df.shape[0] + 1)
df['Class'] = [1 if landing_status ==
               'Success' else 0 for landing_status in df['Booster_Landing']]
df['Year'] = pd.to_datetime(df['Date']).dt.year

# print(df['Class'].value_counts())

colors = {
    'background': '#111111',
    'text': '#becdce',
    'light': '#e7e7e7',
    'dark': '#515151',
    'lineColor': '#687879',
    'gridColor': '#384d4f',
    'br_gridColor': '#454545'
}

binary_class_palette = ['#DE3163', '#50C878']

pieChartHeight = 370

# prepare launch sites options for selectbox

def getLaunchSitesOptions():
    launchSitesOptions = [{"label": "All Launch Sites", "value": "All"}]
    for launchSite in df['Launch_Site'].unique():
        launchSitesOptions.append({"label": launchSite, "value": launchSite})

    return launchSitesOptions


def getBoosterVersions():
    boosterVersionOptions = [{'label': 'All Booster Versions', 'value': 'All'}]
    for boosterVersion in df['Version_Booster'].unique():
        boosterVersionOptions.append(
            {'label': boosterVersion, 'value': boosterVersion})

    return boosterVersionOptions


def getOrbitTypeOptions():
    orbitTypeOptions = [{'label': 'All Orbit types', 'value': 'All'}]

    for orbitType in df['Orbit'].unique():
        if orbitType is None:
            continue
        orbitTypeOptions.append({'label': orbitType, 'value': orbitType})

    return orbitTypeOptions

app = Flask(__name__)

@app.context_processor
def inject_now():
    return {'now': datetime.datetime.utcnow()}

# Falcon 9 Analysis app
falcon_dash_app = Dash(external_stylesheets=[dbc.themes.SOLAR, dbc.icons.BOOTSTRAP],  server=app, routes_pathname_prefix='/falcon9_dashboard/')
falcon_dash_app.title = "Falcon 9"

@falcon_dash_app.callback(
    Output(component_id="success_launchsite_pie_chart",
           component_property="figure"),
    Input(component_id="launch_site", component_property="value"),
    Input('flight_number', 'value'),
    Input('payload_mass', 'value'),
    Input('booster_version', 'value'),
    Input('orbit_type', 'value'),
    Input('year', 'value'),
)
def getSuccessPieChart(launch_site, flight_number, payload_mass, booster_version, orbit_type, year):
    filtered_df = df[((df['Flight_No'] >= flight_number[0]) & (df['Flight_No'] <= flight_number[1]))
                     & ((df['Payload_Mass'] >= payload_mass[0]) & (df['Payload_Mass'] <= payload_mass[1]))
                     & ((df['Year'] >= year[0]) & (df['Year'] <= year[1]))]

    chart_title = "Success landing percentage by Launch Site"

    # filter by booster versoin
    if booster_version != "All":
        filtered_df = df[df['Version_Booster']
                         == booster_version]
    # filter by Orbit type
    if orbit_type != 'All':
        filtered_df = df[df['Orbit'] == orbit_type]

    filtered_df = filtered_df[['Launch_Site', 'Class']]

    if launch_site == None or launch_site == "All":
        fig = px.pie(filtered_df,
                     values='Class',
                     names='Launch_Site',
                     title=chart_title,
                     color_discrete_sequence=px.colors.qualitative.Antique,
                     hole=.3)

    else:
        filtered_df = filtered_df[filtered_df['Launch_Site'] ==
                                  launch_site].value_counts().to_frame().reset_index()

        chart_title = "Landing outcome for Launch Site %s" % launch_site

        # set color sequence for single output value
        color_sequence = binary_class_palette

        if len(filtered_df['Class']) > 0:
            if len(filtered_df['Class']) == 1:
                color_sequence = [
                    binary_class_palette[filtered_df['Class'][0]]]
            else:
                color_sequence = [binary_class_palette[filtered_df['Class']
                                                       [0]], binary_class_palette[filtered_df['Class'][1]]]

        filtered_df['Class'] = filtered_df['Class'].replace(
            {0: 'Failure', 1: 'Success'})

        # print(filtered_df)
        fig = px.pie(filtered_df,
                     values='count',
                     names='Class',
                     title=chart_title,
                     color_discrete_sequence=color_sequence,
                     hole=.3,)

    fig = updateChartLayout(filtered_df, fig, chart_title, pieChartHeight)

    return fig


@falcon_dash_app.callback(
    Output('success_orbit_pie_chart', 'figure'),
    Input('orbit_type', 'value'),
    Input("launch_site", "value"),
    Input('flight_number', 'value'),
    Input('payload_mass', 'value'),
    Input('booster_version', 'value'),
    Input('year', 'value'),
)
def getSuccessOrbitPieChart(orbit_type, launch_site, flight_number, payload_mass, booster_version, year):
    filtered_df = df[~df['Orbit'].isnull()]
    filtered_df = df[((df['Flight_No'] >= flight_number[0]) & (df['Flight_No'] <= flight_number[1]))
                     & ((df['Payload_Mass'] >= payload_mass[0]) & (df['Payload_Mass'] <= payload_mass[1]))
                     & ((df['Year'] >= year[0]) & (df['Year'] <= year[1]))]

    # filter by Booster Version
    if booster_version != 'All':
        filtered_df = df[df['Version_Booster'] == booster_version]

    # filter by Launch Site
    if launch_site != None and launch_site != 'All':
        filtered_df = df[df['Launch_Site'] == launch_site]

    filtered_df['Orbit'] = filtered_df['Orbit'].replace(
        'Ballistic lunar transfer (BLT)', 'BLT')

    filtered_df = filtered_df[['Orbit', 'Class']]

    chart_title = "Success landing percentage for all Orbit types"

    if orbit_type == 'All':
        fig = px.pie(
            filtered_df,
            values='Class',
            names='Orbit',
            color_discrete_sequence=px.colors.qualitative.Antique,
            hole=.3)
    else:
        filtered_df = filtered_df[filtered_df['Orbit'] == orbit_type
                                  ].value_counts().to_frame().reset_index()

        chart_title = "Landing Outcome for Orbit %s" % orbit_type

        # set color sequence for single output value
        color_sequence = binary_class_palette

        if len(filtered_df['Class']) > 0:
            if len(filtered_df['Class']) == 1:
                color_sequence = [
                    binary_class_palette[filtered_df['Class'][0]]]
            else:
                color_sequence = [binary_class_palette[filtered_df['Class']
                                                       [0]], binary_class_palette[filtered_df['Class'][1]]]

        filtered_df['Class'] = filtered_df['Class'].replace(
            {0: 'Failure', 1: 'Success'}, inplace=True)

        fig = px.pie(filtered_df,
                     values='count',
                     names='Class',
                     title=chart_title,
                     color_discrete_sequence=color_sequence,
                     hole=.3,)
    #print('len df 2: \n', filtered_df)
    fig = updateChartLayout(filtered_df, fig, chart_title, pieChartHeight)

    return fig


@falcon_dash_app.callback(
    Output('success_boosterversion_pie_chart', 'figure'),
    Input(component_id="launch_site", component_property="value"),
    Input('flight_number', 'value'),
    Input('payload_mass', 'value'),
    Input('booster_version', 'value'),
    Input('orbit_type', 'value'),
    Input('year', 'value'),
)
def getSuccessRateBoosterVersionPieChart(launch_site, flight_number, payload_mass,  booster_version, orbit_type, year):
    filtered_df = df[((df['Flight_No'] >= flight_number[0]) & (df['Flight_No'] <= flight_number[1]))
                     & ((df['Payload_Mass'] >= payload_mass[0]) & (df['Payload_Mass'] <= payload_mass[1]))
                     & ((df['Year'] >= year[0]) & (df['Year'] <= year[1]))]

    # filrer by Launch Site
    if launch_site != None and launch_site != 'All':
        filtered_df = filtered_df[filtered_df['Launch_Site'] == launch_site]

    # filter by Orbit type
    if orbit_type != 'All':
        filtered_df = df[df['Orbit'] == orbit_type]

    filtered_df = filtered_df[['Version_Booster', 'Class']]

    chart_title = "Successful landings by Booster Version"

    if booster_version == 'All':
        fig = px.pie(
            filtered_df,
            values='Class',
            names='Version_Booster',
            title=chart_title,
            color_discrete_sequence=px.colors.qualitative.Antique,
            hole=.3)
    else:
        filtered_df = filtered_df[filtered_df['Version_Booster'] ==
                                  booster_version].value_counts().to_frame().reset_index()

        # set color sequence for single output value
        color_sequence = binary_class_palette

        if len(filtered_df['Class']) > 0:
            if len(filtered_df['Class']) == 1:
                color_sequence = [
                    binary_class_palette[filtered_df['Class'][0]]]
            else:
                color_sequence = [binary_class_palette[filtered_df['Class']
                                                       [0]], binary_class_palette[filtered_df['Class'][1]]]

        chart_title = "Landing outcome for Booster Version %s" % booster_version

        filtered_df['Class'] = filtered_df['Class'].replace(
            {0: 'Failure', 1: 'Success'})

        fig = px.pie(filtered_df,
                     values='count',
                     names='Class',
                     color_discrete_sequence=color_sequence,
                     hole=.3,)

    fig = updateChartLayout(filtered_df, fig, chart_title, pieChartHeight)

    return fig


@falcon_dash_app.callback(
    Output(component_id='launch_site_v_payload_mass',
           component_property='figure'),
    Input(component_id="launch_site", component_property="value"),
    Input('flight_number', 'value'),
    Input('payload_mass', 'value'),
    Input('booster_version', 'value'),
    Input('orbit_type', 'value'),
    Input('year', 'value'),
)
def getLaunchSiteVsPayloadMass(launch_site, flight_number, payload_mass, booster_version, orbit_type, year):
    filtered_df = df[((df['Flight_No'] >= flight_number[0]) & (df['Flight_No'] <= flight_number[1]))
                     & ((df['Payload_Mass'] >= payload_mass[0]) & (df['Payload_Mass'] <= payload_mass[1]))
                     & ((df['Year'] >= year[0]) & (df['Year'] <= year[1]))]

    chart_title = "Landing outcome of Launch Site for against Payload Mass"

    # filter by booster versoin
    if booster_version != "All":
        filtered_df = df[df['Version_Booster']
                         == booster_version]

    # filter by Orbit type
    if orbit_type != 'All':
        filtered_df = df[df['Orbit'] == orbit_type]

    if launch_site == None or launch_site == "All":
        fig = px.scatter(filtered_df, x='Payload_Mass', y='Class', color='Launch_Site',
                         labels={
                             "Payload_Mass": "Payload Mass (kg)",
                             "Launch_Site": "Launch Site",
                             "Class": "Landing Outcome"
                         },
                         color_discrete_sequence=px.colors.qualitative.Dark2,
                         category_orders={'Class': ['Fail', 'Success']}
                         )
    else:
        filtered_df = filtered_df[filtered_df['Launch_Site'] == launch_site]
        fig = px.scatter(filtered_df, x='Payload_Mass', y='Class', color='Launch_Site',
                         labels={
                             "Payload_Mass": "Payload Mass (kg)",
                             "Launch_Site": "Launch Site",
                             "Class": "Landing Outcome"
                         },
                         color_discrete_sequence=px.colors.qualitative.Dark2,
                         category_orders={'Class': ['Fail', 'Success']}
                         )

        chart_title = "Landing outcome of Launch Site %s against Payload Mass" % (
            launch_site)

    fig.update_xaxes(zeroline=False, linecolor=colors['lineColor'],
                     gridcolor=colors['gridColor'], tickvals=[0, 1])
    fig.update_yaxes(linecolor=colors['lineColor'], type='category',
                     gridcolor=colors['gridColor'])

    fig = updateChartLayout(filtered_df, fig, chart_title, 250)

    fig.update_layout(
        paper_bgcolor='rgba(0, 0, 0, 0.2)',
    )

    return fig


# YEARLY TREND CHARTS


@falcon_dash_app.callback(
    Output('success_yearly_trend_linechart', 'figure'),
    Input('year', 'value'),
    Input("launch_site", "value"),
    Input('flight_number', 'value'),
    Input('payload_mass', 'value'),
    Input('booster_version', 'value'),
    Input('orbit_type', 'value')
)
def getYearlySuccessTrendLineChart(year, launch_site, flight_number, payload_mass,  booster_version, orbit_type):
    filtered_df = df[((df['Flight_No'] >= flight_number[0]) & (df['Flight_No'] <= flight_number[1]))
                     & ((df['Payload_Mass'] >= payload_mass[0]) & (df['Payload_Mass'] <= payload_mass[1]))
                     & ((df['Year'] >= year[0]) & (df['Year'] <= year[1]))]

    # filrer by Launch Site
    if launch_site != None and launch_site != 'All':
        filtered_df = filtered_df[filtered_df['Launch_Site'] == launch_site]

    # filter by Orbit type
    if orbit_type != 'All':
        filtered_df = df[df['Orbit'] == orbit_type]

    # filter by Booster Version
    if booster_version != 'All':
        filtered_df = df[df['Version_Booster'] == booster_version]

    chart_title = "Success Landing Yearly Trend %s - %s" % (year[0], year[1])

    filtered_df_line = filtered_df[['Year', 'Class']
                              ].groupby('Year').sum(numeric_only=True).reset_index()
    
    # print(filtered_df)
    fig = px.line(
        filtered_df_line,
        x='Year',
        y='Class',
        color_discrete_sequence=['#50C878'],
        labels={
            'Class': 'Successful Landings'
        }
    )

    fig.update_yaxes(
        zeroline=False, linecolor=colors['lineColor'], 
        gridcolor=colors['gridColor'])
    fig.update_xaxes(zeroline=False, linecolor=colors['lineColor'], type='date',
                     gridcolor=colors['gridColor'])

    fig = updateChartLayout(filtered_df, fig, chart_title, 300)

    fig.update_layout(
        paper_bgcolor='rgba(0, 0, 0, 0.2)',
    )

    return fig


@falcon_dash_app.callback(
    Output('yearly_launches_barchart', 'figure'),
    Input('year', 'value'),
    Input("launch_site", "value"),
    Input('flight_number', 'value'),
    Input('payload_mass', 'value'),
    Input('booster_version', 'value'),
    Input('orbit_type', 'value')
)
def getYearlyLaunchesBarchart(year, launch_site, flight_number, payload_mass,  booster_version, orbit_type):
    filtered_df = df[((df['Flight_No'] >= flight_number[0]) & (df['Flight_No'] <= flight_number[1]))
                     & ((df['Payload_Mass'] >= payload_mass[0]) & (df['Payload_Mass'] <= payload_mass[1]))
                     & ((df['Year'] >= year[0]) & (df['Year'] <= year[1]))]

    # filrer by Launch Site
    if launch_site != None and launch_site != 'All':
        filtered_df = filtered_df[filtered_df['Launch_Site'] == launch_site]

    # filter by Orbit type
    if orbit_type != 'All':
        filtered_df = df[df['Orbit'] == orbit_type]

    # filter by Booster Version
    if booster_version != 'All':
        filtered_df = df[df['Version_Booster'] == booster_version]

    chart_title = "Yearly Launches %s - %s" % (year[0], year[1])

    filtered_df_bar = filtered_df[['Year']].value_counts().reset_index().sort_values(by="Year", ascending=True)
    
    filtered_df_line = filtered_df[['Year', 'Class']
                              ].groupby('Year').sum(numeric_only=True).sort_values(by="Year", ascending=True).reset_index()

    df_combined = filtered_df_bar.merge(filtered_df_line, on="Year")
    
    #print(df_combined)
    fig = px.bar(
        df_combined,
        x='Year',
        y='count',
        text_auto='.2s',
        color_discrete_sequence=[px.colors.qualitative.Prism[3]],
        labels={
            'count': 'Number of Launches'
        }
    ).add_traces(
        px.line(
            df_combined,
            x='Year',
            y='Class',
            color_discrete_sequence=['#FFBF00'],
            labels={
                'Class': 'Successful Landings'
            }
        ).update_traces(showlegend=True, name="success").data
    )

    fig.update_yaxes(
        zeroline=False, linecolor=colors['lineColor'], gridcolor=colors['gridColor'])
    fig.update_xaxes(zeroline=False, linecolor=colors['lineColor'], type='date',
                     gridcolor=colors['gridColor'], categoryorder='category ascending')

    fig = updateChartLayout(filtered_df, fig, chart_title, 300)
    fig.update_layout(
        paper_bgcolor='rgba(0, 0, 0, 0.2)',
    )

    return fig


@falcon_dash_app.callback(
    Output('launch_site', 'value'),
    Output('flight_number', 'value'),
    Output('payload_mass', 'value'),
    Output('booster_version', 'value'),
    Output('orbit_type', 'value'),
    Output('year', 'value'),
    Input('reset_button', 'n_clicks'),
    State('launch_site', 'value')
)
def resetFilters(nclicks, launch_site):
    return 'All', [df['Flight_No'].min(), df['Flight_No'].max()], [df['Payload_Mass'].min(), df['Payload_Mass'].max()], 'All', 'All', [df['Year'].min(), df['Year'].max()]


def updateChartLayout(filtered_df, fig, chart_title, height):
    if len(filtered_df) == 0:
        fig.update_layout(
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font_color=colors['text'],
            title_text=chart_title,
            title_x=0.5,
            autosize=False,
            height=height,
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[
                {
                    "text": "No data available for this graph.",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {
                        "size": 14
                    }
                }
            ]
        )

    else:
        fig.update_layout(
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font_color=colors['text'],
            title_text=chart_title,
            title_x=0.5,
            autosize=False,
            height=height
        )

    return fig

falcon_dash_app.layout = html.Div(
    [
        dbc.Container(
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Div(
                                            html.Img(
                                                src="/static/assets/images/Falcon_9_logo.png",
                                                width=100,
                                                style={
                                                    'margin-bottom': '20px',
                                                })
                                        ),
                                        width=3
                                    ),

                                    dbc.Col(
                                        html.Div(
                                            [
                                                html.H2("SpaceX Falcon 9", style={
                                                        'color': colors['light']}),
                                                html.H5(
                                                    "Stage 1 Landing Analysis", style={
                                                        'color': colors['light']})
                                            ]
                                        ),
                                        width=9,
                                        align='center',
                                    )

                                ]
                            ),

                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    dbc.Button(
                                                        [
                                                            html.I(className="bi bi-house-fill me-2"),
                                                            "Back to Home"
                                                        ], 
                                                        href="javascript:history.back()",
                                                        id='back_button', 
                                                        color="warning", 
                                                        className="me-1", 
                                                        n_clicks=0)
                                                ]
                                            ),
                                            
                                            html.Hr(),
                                            
                                            dbc.Label("Launch Site"),
                                            dcc.Dropdown(
                                                id='launch_site',
                                                options=[{'label': ls, 'value': ls}
                                                         for ls in df['Launch_Site'].unique()] + [{'label': 'All Launch Sites', 'value': 'All'}],
                                                value="All",
                                                placeholder="Select Launch Site",
                                                style={
                                                    'background': colors['text'],
                                                    'color': '#333333',
                                                    'border-radius': '4px',
                                                }
                                            ),

                                            html.Hr(),

                                            dbc.Label("Flight Number"),
                                            dcc.RangeSlider(
                                                id="flight_number",
                                                min=0, max=df['Flight_No'].max() + 1, step=25,
                                                value=[df['Flight_No'].min(),
                                                       df['Flight_No'].max()],
                                                allowCross=False,
                                                tooltip={
                                                    "placement": "bottom", "always_visible": False},
                                                marks={fn: {'label': str(fn), 'style': {'color': str(
                                                    colors['text'])}} for fn in list(range(0, df['Flight_No'].max() + 1, 25))}
                                            ),

                                            html.Hr(),

                                            dbc.Label("Payload Mass (kg)"),
                                            dcc.RangeSlider(
                                                id="payload_mass",
                                                min=0, max=df['Payload_Mass'].max() + 1, step=1000,
                                                value=[df['Payload_Mass'].min(),
                                                       df['Payload_Mass'].max()],
                                                allowCross=False,
                                                tooltip={
                                                    "placement": "bottom", "always_visible": False},
                                                marks={fn: {'label': str(fn), 'style': {'color': str(
                                                    colors['text'])}} for fn in list(range(0, round(df['Payload_Mass'].max()) + 1, 2000))}
                                            ),

                                            html.Hr(),

                                            dbc.Label("Booster Version"),
                                            dcc.Dropdown(
                                                id="booster_version",
                                                options=[{'label': bv, 'value': bv}
                                                         for bv in df['Version_Booster'].unique()] + [{'label': 'All Booster Versions', 'value': 'All'}],
                                                value="All",
                                                searchable=True,
                                                placeholder="Select a Booster Version",
                                                style={
                                                    'background': colors['text'],
                                                    'color': '#333333',
                                                    'border-radius': '4px',
                                                }
                                            ),

                                            html.Hr(),

                                            dbc.Label("Orbit"),
                                            dcc.Dropdown(
                                                id="orbit_type",
                                                options=[{'label': orbit, 'value': orbit}
                                                         for orbit in df['Orbit'].unique()] + [{'label': 'All Orbit types', 'value': 'All'}],
                                                value="All",
                                                searchable=True,
                                                placeholder="Select an Orbit type",
                                                style={
                                                    'background': colors['text'],
                                                    'color': '#333333',
                                                    'border-radius': '4px',
                                                }
                                            ),

                                            html.Hr(),

                                            dbc.Label("Year"),
                                            dcc.RangeSlider(
                                                id="year",
                                                min=df['Year'].min(), max=df['Year'].max(), step=1,
                                                allowCross=False,
                                                tooltip={
                                                    "placement": "bottom", "always_visible": False},
                                                marks={
                                                    yr: {'label': str(yr), 'style': {
                                                        'color': str(colors['text'])}}
                                                    for yr in list(range(df['Year'].min(), df['Year'].max() + 1, 2))
                                                },
                                                value=[df['Year'].min(),
                                                       df['Year'].max()],
                                            ),

                                            html.Hr(),

                                            dbc.Button(
                                                "Reset Filters", id='reset_button', color="primary", className="me-1", n_clicks=0)

                                        ]
                                    ),
                                ]
                            )

                        ],

                        width=3),

                    dbc.Col(

                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Div(
                                            dcc.Graph(
                                                id="success_launchsite_pie_chart")
                                        ),
                                        width=4
                                    ),

                                    dbc.Col(
                                        html.Div(
                                            dcc.Graph(
                                                id="success_boosterversion_pie_chart")
                                        ),
                                        width=4
                                    ),

                                    dbc.Col(
                                        html.Div(
                                            dcc.Graph(
                                                id="success_orbit_pie_chart")
                                        ),
                                        width=4
                                    )
                                ]
                            ),

                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Div(
                                            dcc.Graph(
                                                id="launch_site_v_payload_mass")
                                        ),
                                    )
                                ],

                                className='mb-2'
                            ),

                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Div(
                                            dcc.Graph(
                                                id="success_yearly_trend_linechart")
                                        ),
                                        width=6
                                    ),

                                    dbc.Col(
                                        html.Div(
                                            dcc.Graph(
                                                id="yearly_launches_barchart")
                                        ),
                                        width=6
                                    )
                                ]
                            )

                        ],

                        width=9)
                ]
            ),
            fluid=True
        )

    ],
    style={
        'padding': 10,
        'color': colors['text'],
    }
)

# CO2 emission Analysis app
co2_emission_dash_app = Dash(external_stylesheets=[dbc.themes.DARKLY, dbc.icons.BOOTSTRAP],  server=app, routes_pathname_prefix='/co2_emission_dashboard/')

co2_all_countries = pd.read_csv('ML_models/co2_emission_analysis/all_countries.csv')
co2_geo_region_df = pd.read_csv('ML_models/co2_emission_analysis/geo_regions.csv')
co2_economy_region_df = pd.read_csv('ML_models/co2_emission_analysis/economy_groups.csv')

geo_json_r = open('ML_models/co2_emission_analysis/world-countries.json', 'r')
geo_json = geo_json_r.read()

@co2_emission_dash_app.callback(
    Output('co2_map_div', 'figure'),
    Input('co2_year', 'value')
)
def co2_emissionr_map(co2_year):
    year_to_display = co2_year[1]
    chart_title = f"Carbon dioxide emission by country in {year_to_display} (kilotons)"
    df_all_countries_filter_by_year = co2_all_countries[co2_all_countries['year'] == year_to_display]
    #df_all_countries_filter_by_year

    fig = px.choropleth(df_all_countries_filter_by_year, locations="country_code",
                    color="value", # lifeExp is a column of gapminder
                    hover_name="country_name", # column to add to hover information
                    color_continuous_scale=px.colors.sequential.Reds,
                    animation_frame="year",
                    labels={
                        'value': 'CO₂',
                        'country_name': 'Country',
                        'country_code': 'Country',
                        'year': 'Year'
                    })
    
    fig = updateChartLayout(df_all_countries_filter_by_year, fig, chart_title, 375)
    fig.update_yaxes(
        zeroline=False, linecolor=colors['lineColor'], gridcolor=colors['br_gridColor'])
    fig.update_geos(
        resolution=50, showocean=True, oceancolor="#222",
        showcountries=True, countrycolor="RebeccaPurple"
    )
    fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
    
    return fig

@co2_emission_dash_app.callback(
    Output('co2_top_5_contributors_bar_chart', 'figure'),
    Input('co2_year', 'value')
)
def top_5_contributors(year_range):
   
    df_top_5 = co2_all_countries[((co2_all_countries['year'] >= year_range[0]) & (co2_all_countries['year'] <= year_range[1]))].groupby('country_name').sum(numeric_only=True).sort_values(by='value', ascending=False).head(5)
    
    chart_title = f"Top 5 contributors to CO₂ emission ({year_range[0]}-{year_range[1]})"
    
    fig = px.bar(
            df_top_5, 
            x=df_top_5.index, 
            y='value', 
            text_auto='.2s',
            color='value',
            color_continuous_scale=px.colors.sequential.Brwnyl,
            labels={
                'value': 'CO₂ (kilotons)',
                'country_name': 'Country'
            })
    
    fig = updateChartLayout(df_top_5, fig, chart_title, 375)
    fig.update_yaxes(
        zeroline=False, linecolor=colors['lineColor'], gridcolor=colors['br_gridColor'])
    fig.update_xaxes(zeroline=False, linecolor=colors['lineColor'],
                     gridcolor=colors['br_gridColor'])
    
    fig.update_layout(
        paper_bgcolor='rgba(0, 0, 0, 0.2)',
    )
    
    return fig

@co2_emission_dash_app.callback(
    Output('co2_geo_region_line_chart', 'figure'),
    Input('co2_year', 'value')
)
def co2_geo_region_line_graph(year_range):
    
    df_filtered = co2_geo_region_df[((co2_geo_region_df['year'] >= year_range[0]) & (co2_geo_region_df['year'] <= year_range[1]))]
    
    chart_title = f"Different geographical regions between {year_range[0]} and {year_range[1]}"
    
    if (year_range[1] - year_range[0]) > 5:
        fig = px.line(
            df_filtered, 
            x='year', 
            y='value',
            color='country_name',
            labels={
                'value': 'CO₂ emission (kilotons)',
                'year': 'Year',
                'country_name': 'Geographical region'
            },
            color_discrete_sequence=px.colors.qualitative.Bold)
        
        fig.update_yaxes(
        zeroline=False, linecolor=colors['lineColor'], gridcolor=colors['br_gridColor'])
        fig.update_xaxes(zeroline=False, linecolor=colors['lineColor'], 
                        gridcolor=colors['br_gridColor'], type='date', categoryorder='category ascending')
    else:
        fig = px.bar(
            df_filtered, 
            x='value', 
            y='year', 
            orientation='h',
            text_auto='.2s',
            color='country_name',
            color_discrete_sequence=px.colors.qualitative.Antique,
            labels={
                'value': 'CO₂ (kilotons)',
                'country_name': 'Geographical region'
            })
        
        fig.update_yaxes(
        zeroline=False, linecolor=colors['lineColor'], gridcolor=colors['br_gridColor'], type='date', categoryorder='category ascending')
        fig.update_xaxes(zeroline=False, linecolor=colors['lineColor'], 
                        gridcolor=colors['br_gridColor'])
    
    fig = updateChartLayout(df_filtered, fig, chart_title, 375)
            
    fig.update_layout(
        paper_bgcolor='rgba(0, 0, 0, 0.2)',
    )
    
    return fig

@co2_emission_dash_app.callback(
    Output('co2_top_5_contributors_all_time_pie_chart', 'figure'),
    Input('co2_year', 'value')
)
def top_5_contributors_all_time(year_range):
    df_total_co2 = co2_all_countries.groupby('country_name').sum(numeric_only=True).sort_values(by='value', ascending=False).head(5).reset_index()
    #print(df_total_co2)
    chart_title = "Total CO₂ emission by top contributors all time"
    
    fig = px.pie(
            df_total_co2,
            values='value',
            names='country_name',
            hole=0.3,
            color_discrete_sequence=px.colors.qualitative.Antique,
            labels={
                'value': 'CO₂ (kilotons)',
                'country_name': 'Country'
            }
        )
    
    fig = updateChartLayout(df_total_co2, fig, chart_title, 375)
    
    fig.update_layout(autosize=False)
        
    return fig

@co2_emission_dash_app.callback(
    Output('co2_economy_region_line_chart', 'figure'),
    Input('co2_year', 'value')
)
def co2_economy_region(year_range):
    
    chart_title = f"Economy groups between {year_range[0]} and {year_range[1]}"
    
    df_filtered = co2_economy_region_df[((co2_economy_region_df['year'] >= year_range[0]) & (co2_economy_region_df['year'] <= year_range[1]))]
    
    if (year_range[1] - year_range[0]) > 5:
        fig = px.line(
            df_filtered, 
            x='year', 
            y='value',
            color='country_name',
            labels={
                'value': 'CO₂ emission (kilotons)',
                'year': 'Year',
                'country_name': 'Economy group'
            },
            color_discrete_sequence=px.colors.qualitative.Vivid)
    else:
        fig = px.bar(
            df_filtered, 
            x='year', 
            y='value', 
            text_auto='.2s',
            color='country_name',
            color_discrete_sequence=px.colors.sequential.Brwnyl_r,
            labels={
                'value': 'CO₂ (kilotons)',
                'country_name': 'Economy group'
            })
    
    fig = updateChartLayout(df_filtered, fig, chart_title, 375)
    
    fig.update_yaxes(
        zeroline=False, linecolor=colors['lineColor'], gridcolor=colors['br_gridColor'])
    fig.update_xaxes(zeroline=False, linecolor=colors['lineColor'], 
                     gridcolor=colors['br_gridColor'], type='date', categoryorder='category ascending')  
        
    fig.update_layout(
        paper_bgcolor='rgba(0, 0, 0, 0.2)',
    )
    
    return fig

co2_emission_dash_app.title = "CO₂ emission"
co2_emission_dash_app.layout = html.Div(
    [
        dbc.Container(
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div(
                                    [
                                        html.Img(
                                            src="/static/assets/images/co2_dash_logo.png",
                                            width=120)                                           
                                    ]
                                ),
                                width='auto'
                            ),
                        
                            dbc.Col(
                                html.Div(
                                    [
                                        html.H2(
                                            [
                                                "CO₂ Emission ",
                                                html.Small('(Worldwide Analysis)', style={
                                                    'font-size': '24px'
                                                })
                                            ], style={
                                                'color': colors['text'],
                                                'display': 'inline'
                                            }
                                        ),
                                    ]
                                ),
                                width='auto'
                            ),
                            dbc.Col(
                                html.Div(
                                    [
                                        dbc.Button(
                                            [
                                                html.I(className="bi bi-house-fill me-2"),
                                                "Back to Home"
                                            ], 
                                            href="javascript:history.back()",
                                            id='back_button', 
                                            color="danger", 
                                            className="me-1", 
                                            n_clicks=0)
                                    ],
                                    style={'text-align': 'right'}
                                ),
                            )
                        ],
                        className='mb-2',
                        align="center"
                    ),
                    html.Hr(),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            dcc.Graph(id="co2_map_div"),
                                        ]                          
                                    ),
                                ],
                                className="col col-md-12 col-lg-12 col-xl-5"
                            ),
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            dcc.Graph(id="co2_top_5_contributors_bar_chart")
                                        ]                           
                                    ),
                                ],
                                className="col col-md-6 col-lg-6 col-xl-4"
                            ),
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            dcc.Graph(id="co2_top_5_contributors_all_time_pie_chart")
                                        ]                           
                                    ),
                                ],
                                className="col col-md-6 col-lg-6 col-xl-3"
                            )
                        ],
                        className='mb-2'
                    ), 
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            dcc.Graph(id="co2_geo_region_line_chart")
                                        ]                           
                                    ),
                                ],
                                className="col col-md-12 col-lg-6"
                            ),
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            dcc.Graph(id="co2_economy_region_line_chart")
                                        ]                           
                                    ),                                    
                                ],
                                className="col col-md-12 col-lg-6"
                            )
                        ],
                        className='mb-3'
                    ), 
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.H6([
                                        html.I(className="bi bi-sliders me-2"),
                                        'Timeline'
                                        ], style={'text-align': 'center'}),
                                    dcc.RangeSlider(
                                        id="co2_year",
                                        min=co2_all_countries['year'].min(), max=co2_all_countries['year'].max(), step=1,
                                        tooltip={"placement": "top", "always_visible": False},
                                        marks={
                                            yr: {'label': str(yr), 'style': {
                                                'color': str(colors['text']), 'font-size': '16px', 'padding-bottom': 35}}
                                            for yr in list(range(co2_all_countries['year'].unique().min(), co2_all_countries['year'].unique().max() + 1, 10))
                                        },
                                        value=[co2_all_countries['year'].unique().min(), co2_all_countries['year'].unique().max()],
                                        allowCross=False
                                    )
                                ]
                            )
                        ],
                        style={'position': 'fixed', 'width': '100%', 'bottom': '0px', 'background': '#222222'}
                    )                   
                ]
            ),
            fluid=True
        ),
        
    ],
    style={
        'padding': 10,
        'padding-bottom': 50,
        'color': colors['text'],
    }
)

# Recommender system
recommender_system_df = pd.read_csv('https://query.data.world/s/uikepcpffyo2nhig52xxeevdialfl7')
recommender_system_df_orig = recommender_system_df[['Title', 'Year', 'Genre','Director','Actors','Plot', 'Poster', 'imdbRating']].set_index('Title')
recommender_system_df = recommender_system_df[['Title', 'Year', 'Genre','Director','Actors','Plot', 'Poster', 'imdbRating']]

def recomender_preprocessing(movie_df):
    # data preprocessing for recomender system
    # set lowercase and split on commas
    movie_df['Actors'] = movie_df['Actors'].map(lambda x: x.lower().split(','))
    movie_df['Genre'] = movie_df['Genre'].map(lambda x: x.lower().split(','))
    movie_df['Director'] = movie_df['Director'].map(lambda x: x.lower().split(' '))

    # join Actors and Director names as a single string
    movie_df['Actors'] = movie_df.apply(lambda row: [x.replace(' ', '') for x in row['Actors']], axis=1)
    movie_df['Director'] = movie_df.apply(lambda row: ''.join(row['Director']), axis=1)
    # for index, row in movie_df.iterrows():
    #     #print(index)
    #     row['Actors'] = [x.replace(' ', '') for x in row['Actors']]
    #     row['Director'] = ''.join(row['Director'])
        
    #     movie_df.loc[:, ('Actors', index)] = row['Actors']
    #     movie_df.loc[:, ('Director', index)] = row['Director']
    
    return movie_df

recommender_system_df = recomender_preprocessing(recommender_system_df)

# Extract keywords using Rake()

def extract_keywords(movie_df):
    movie_df['Key_words'] = "" # initialize the column for storing keywords
    
    for index, row in movie_df.iterrows():
        plot = row['Plot']
        
        # initiate Rake
        r = Rake()
        
        # extracting the words by passing the text
        r.extract_keywords_from_text(plot)
        
        # preparing a dictionary with keywords and their scores
        keyword_dict_score = r.get_word_degrees()
        
        # assign keywords to the new column
        row['Key_words'] = list(keyword_dict_score.keys())
       
    # we do not need 'Plot' column anymore
    movie_df.drop('Plot', axis=1, inplace=True)
    
    # set Title as index, then we can easily identify records rather than using numerical indices
    movie_df.set_index('Title', inplace=True)
    
    return movie_df 

recommender_system_df = extract_keywords(recommender_system_df)

# create bag of words
def create_bag_of_words(movie_df):
    # initialize bag_of_words column
    movie_df['bag_of_words'] = ""
    
    columns = movie_df.columns
    # for index, row in movie_df.iterrows():
    #     words = ''
    #     for col in columns:
    #         if col == 'Director':
    #             #print(row[col])
    #             words += row[col] + ' '
    #         elif col == 'Actors':
    #             words += ' '.join(row[col]) + ' '
    #     movie_df.loc[:, ('bag_of_words', index)] = words
    
    movie_df['bag_of_words'] = movie_df.apply(
        lambda row: bag_of_words_row(row['Actors'], row['Director']),
        axis=1
    )
        
    # now we only need the index and the bag_of_words column. So, we drop other columns
    keep_cols = ['bag_of_words', 'Title', 'Year', 'Director', 'Poster', 'imdbRating']
    movie_df.drop([col for col in columns if col not in keep_cols], axis=1, inplace=True)
    
    return movie_df

def bag_of_words_row(actors, director):
    words = ''
    words += director + ' '
    words += ' '.join(actors) + ' '
    
    return words

recommender_system_df = create_bag_of_words(recommender_system_df)
#print(recommender_system_df[['Director', 'bag_of_words']])
#print(recommender_system_df)

# apply Countervectorizer
# this tokenize the words by counting the frequesncy. This is needed for calculate Cosine similarity
# after that in the same function cosine_similarity finction is also applied and return cosine_sim matrix
def count_vectorizer(df):
    count_vector = CountVectorizer()
    count_matrix = count_vector.fit_transform(df['bag_of_words'])
    
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    
    return cosine_sim

cosine_sim = count_vectorizer(recommender_system_df)

# implement recomender function
indeces = pd.Series(recommender_system_df.index)

def recommender(title, cosine_sim=cosine_sim):
    recommendations = []
    
    # get relevant indeces
    if len(indeces[indeces == title]) > 0:
        search_idx = indeces[indeces == title].index[0]
    else:
        return []
    
    similarities = pd.Series(cosine_sim[search_idx]).sort_values(ascending=False)
    
    # get top 10 matches (indexes)
    # use this indexes again to retrieve movie titles
    top_10_matches = list(similarities.iloc[1:11].index)
    print(top_10_matches)
    
    # store best matched titles
    for i in top_10_matches:
        recommendations.append(indeces[i])
        
    return recommendations

# load saved model files
mood_class_strings = ['Sad', 'Happy']
mood_detection_model = load_model(
    'ML_models/happy_model.keras')
with open('ML_models/lr_banknotes_model.pkl', 'rb') as f:
    banknote_model = pickle.load(f)

with open('ML_models/Blackfriday_DT_model.pkl', 'rb') as f:
    black_friday_dt_model = pickle.load(f)

# this list is not essential. We can use model output index as the result. But for the consistancy of the program I'm using it here.
sign_language_class_strings = [0, 1, 2, 3, 4, 5]
sign_language_model = load_model(
    'ML_models/sign_laguange.keras')

sign_language_resnet_model = load_model(
    'ML_models/sign_language_resnet50')

alpaca_mobilenetv2_model = load_model(
    'ML_models/alpaca_mobile_netv2')

co2_emission_lstm_model = load_model('ML_models/co2_emission_lstm')
co2_emission_scalerfile = 'ML_models/co2_emission_lstm/scaler.sav'
co2_emission_scaler = pickle.load(open(co2_emission_scalerfile, 'rb'))
co2_orig_csv = pd.read_csv('ML_models/co2_emission_lstm/year_emission.csv')

up_path = os.path.join(os.getcwd(), 'static', 'assets', 'temp')

UPLOAD_FOLDER = up_path
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

my_work = [
    {
        'header': 'End-to-end',
        'application_url': 'black_friday',
        'image': 'black_friday.png',
        'title': 'Black Friday Purchase Prediction',
        'description': "This project will understand the customer purchase behaviour (specifically, purchase amount) against various products of different categories. They have shared purchase summary of various customers for selected high volume products from last month.",
        'github': 'https://github.com/tharangachaminda/Black_Friday_Purchase',
        'icons': ['python', 'jupyterlab', 'flask', 'aws']
    },

    {
        'header': 'End-to-end',
        'application_url': 'banknotes_authentication',
        'image': 'bank_notes.png',
        'title': 'Banknotes Authentication',
        'description': "Banknote analysis refers to the examination of paper currency to determine its legitimacy and identify potential counterfeits. In this python project, I am trying to build a <b>Classification Machine Learning models</b> to predict banknotes are genuine or forged.",
        'github': 'https://github.com/tharangachaminda/banknotes_analysis',
        'icons': ['python', 'jupyterlab', 'flask', 'heroku']
    },

    {
        'header': 'Analysis',
        'application_url': 'falcon9_dashboard',
        'image': 'falcon_9.png',
        'title': 'SpaceX Falcon 9 1st Stage Landing Prediction',
        'description': "SpaceX re-uses the Stage 1 boosters of Falcon 9 rockets. This project is an analysis for successful Stage 1 landing prediction. I have used SpaceX API and Webscraping for data collection, SQlite database for data storage.",
        'github': 'https://github.com/tharangachaminda/Falcon9_First_stage_Landing',
        'icons': ['python', 'jupyterlab', 'sql', 'sqlite', 'folium', 'plotly-dash', 'heroku']
    },

    {
        'header': 'Recommender System',
        'image': 'netflix.png',
        'title': 'Netflix Recommender System (Popularity Based)',
        'description': "Netflix Recommender system is one of the best recommender systems in the world. In this project I've used <b>movies and rating</b> datasets and <b>NLTK toolkit</b> to build this popularity based recommender system.",
        'github': 'https://github.com/tharangachaminda/popularity-based-recommendation-system',
        'icons': ['Python', 'jupyterlab', 'nltk', 'flask', 'heroku']
    },

    {
        'header': 'Recommender System',
        'application_url': 'recommender_content_based',
        'image': 'netflix.png',
        'title': 'Netflix Recommender System (Content Based)',
        'description': "Netflix Recommender system is one of the best recommender systems in the world. In this project I've used <b>movies datasets</b> and <b>Cosine similarity</b> to build this content based recommender system.",
        'github': 'https://github.com/tharangachaminda/content_based_recommender_system',
        'icons': ['Python', 'jupyterlab', 'nltk', 'flask', 'heroku']
    },

    {
        'header': 'Analysis',
        'image': 'bookstore.png',
        'title': 'Bookstore Web scraping',
        'description': "In this project I have built a mechanism to collect information about every book in the website, scraping through pagination. This project is associated with https://books.toscrape.com/ website which is specially design for training web scraping. ",
        'github': 'https://github.com/tharangachaminda/bookstore_webscraping',
        'icons': ['Python', 'jupyterlab', 'flask', 'heroku']
    },

    {
        'header': 'End-to-end (Deep Learning)',
        'application_url': 'mood_detection',
        'image': 'mood_classifier.png',
        'title': 'Mood Classifier',
        'description': "A mood classification is a type of machine learning task that is used to recognize human moods or emotions. In this project I have implemented a CNN model for recognizing smiling or not smiling humans using Tensorflow Keras Sequential API.",
        'github': 'https://github.com/tharangachaminda/cnn_mood_classifier',
        'icons': ['python', 'jupyterlab', 'tensorflow', 'flask', 'heroku']
    },

    {
        'header': 'End-to-end (Deep Learning - CNN)',
        'application_url': 'sign_language_recognition',
        'image': 'sign_language_digits.png',
        'title': 'Sign Language Digits Recognition',
        'description': "Sing language is a visual-gestural language used by deaf and hard-to-hearing individuals to convey imformation, thoughts and emotions. In this project I have implemented a CNN model for recognizing sign language digits 0 to 5 using Keras Functional API.",
        'github': 'https://github.com/tharangachaminda/cnn_sign_language_detection',
        'icons': ['python', 'jupyterlab', 'tensorflow', 'flask', 'heroku']
    },

    {
        'header': 'End-to-end (Deep Learning - Residual Network)',
        'application_url': 'sign_language_recognition_resnet',
        'image': 'sign_language_digits.png',
        'title': 'Sign Language Digits Recognition',
        'description': "Very deep neural networks suffer from a problem called vanishing/exploding gradients. Deep Residual Learning for Image Recognition resolves theis issue. A Residual Network, also known as ResNet, is a type of deep learning network architecture that introduces the concept of residual learning.",
        'github': 'https://github.com/tharangachaminda/cnn_resnet',
        'icons': ['python', 'jupyterlab', 'tensorflow', 'flask', 'heroku']
    },

    {
        'header': 'End-to-end (Deep Learning - MobileNetV2)',
        'application_url': 'alpaca_mobilenetv2',
        'image': 'alpaca_mobilenetv2.png',
        'title': 'Binary Classification (Transfer Learning)',
        'description': "Transfer Learning in Neural Network is a technique used in machine learning where knowledge gained from training one model (source domain) is transferred and applied to a different but related model (target domain). This involves taking a pre-trained model developed for one task and fine-tuned or using its learned features to solve another related task.",
        'github': 'https://github.com/tharangachaminda/transfer_learning_with_mobilenet_v2',
        'icons': ['python', 'jupyterlab', 'tensorflow', 'flask', 'heroku']
    },
    
    {
        'header': 'End-to-end (Deep Learning - LSTM)',
        'application_url': 'co2_emission_lstm',
        'image': 'co2_emission.png',
        'title': 'CO<sub>2</sub> emission prediction of Sri Lanka (LSTM)',
        'description': "LSTM (Long Short-Term Memory) is a type of Recurrent Neural Network (RNN) architecture designed to efficiently capture and utilize long-term dependencies in <b>sequential data</b>. In this project, I will be developping an LSTM model to predict future CO<sub>2</sub> emission. CO<sub>2</sub> emission data is a Timeseries dataset, which fits for sequential model perfectly.",
        'github': 'https://github.com/tharangachaminda/co2_emission_prediction',
        'icons': ['python', 'jupyterlab', 'tensorflow', 'flask', 'aws']
    },
    
    {
        'header': 'Analysis',
        'application_url': 'co2_emission_dashboard',
        'image': 'co2_emission_analysis.png',
        'title': 'CO<sub>2</sub> emission around the world.',
        'description': "Countries vary significantly in their CO<sub>2</sub> emission levels due to differences in industrialization, energy consumption, transportation, and policies regarding environmental regulations. Some nations, particularly highly industrialized ones like China, the United States, India, and the European Union countries, tend to produce higher CO<sub>2</sub> emissions.",
        'github': 'https://github.com/tharangachaminda/co2_emission_analysis',
        'icons': ['python', 'jupyterlab', 'folium', 'plotly-dash', 'aws']
    },

]


@app.route("/")
def home():
    return render_template('home.html', mywork=my_work)


@app.route("/mood_detection")
def mood_detection():
    return render_template("mood_detection.html")


@app.route("/predict_cnn/<task>", methods=["POST"])
def predict_cnn(task):
    if request.method == 'POST':
        if 'image_file' not in request.files:
            return render_template('model_result.html', message={'type': 'error', 'text': 'You have not uploaded an image.'})

        model_obj = mood_detection_model
        output_class_strings = mood_class_strings
        inputImageWidth = 64
        inputImageHeight = 64
        preprocessing_method = None

        if task == "sign_language":
            model_obj = sign_language_model
            output_class_strings = sign_language_class_strings
        elif task == 'sign_language_resnet':
            model_obj = sign_language_resnet_model
            output_class_strings = sign_language_class_strings
        elif task == 'alpaca_mobilenetv2':
            model_obj = alpaca_mobilenetv2_model
            output_class_strings = ['Not an Alpaca', 'Alpaca']
            inputImageWidth = 160
            inputImageHeight = 160
            preprocessing_method = tf.keras.applications.mobilenet_v2.preprocess_input

        image = request.files['image_file']

        if not allowed_file(image.filename):
            return render_template('model_result.html', message={'type': 'error', 'text': 'Not a valid image file.'})

        file_extension = image.filename.rsplit('.', 1)[1]
        temp_filename = f'{int(time.time())}.{file_extension}'
        image.save(os.path.join(up_path, temp_filename))
        temp_file_path = os.path.join(up_path, temp_filename)
        imageArray = imageToArray(
            temp_file_path, inputImageWidth, inputImageHeight, preprocessing_method)

        if imageArray is False:
            return render_template('model_result.html', message={'type': 'error', 'text': 'Your image has some issues.'})

        model_predict_prob = []
        model_predict_prob = model_obj.predict(imageArray)
        print(model_predict_prob)

        if task == 'sign_language' or task == 'sign_language_resnet':
            model_predict_int = np.argmax(model_predict_prob)
        else:
            model_predict_int = int(model_predict_prob > 0.5)

        print(model_predict_prob, model_predict_int)
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        else:
            print('file path does not exist')

        model_final_output = output_class_strings[model_predict_int]

        return render_template('model_result.html', message={'type': 'normal', 'text': f'Model predicts as <b>"{model_final_output}"</b> with probability {model_predict_prob}'})


@app.route('/sign_language_recognition')
def sign_language_recognition():
    model_summary = sign_language_model.to_json()
    # print(model_summary)
    return render_template("sign_language_detection.html", sign_model={"model_json": model_summary})


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def imageToArray(imageName, inputWidth=64, inputHeight=64, preprocessing_method=None):
    # Load the image and resize it to the desired dimensions
    # image_path = f'images/{imageName}'
    image_path = imageName
    width, height = inputWidth, inputHeight

    image = Image.open(image_path)

    if image.width < inputWidth or image.height < inputHeight:
        return False

    image = image.resize((width, height))
    print(image.width)
    # Convert the image to a NumPy array and normalize the pixel values (if necessary)
    image_array = np.array(image)
    if preprocessing_method is None:
        image_array = image_array / 255.  # Normalize the pixel values between 0 and 1

    # plt.imshow(image_array)
    # plt.show()

    # print(image_array.shape)
    # Reshape the image array to match the input shape of your model
    # Assumes the input shape is (width, height, 3)
    try:
        image_array = image_array.reshape(1, width, height, 3)
        if preprocessing_method is not None:
            image_array = preprocessing_method(image_array)
    except:
        if os.path.exists(image_path):
            os.remove(image_path)
        return False

    return image_array


@app.route('/black_friday')
def black_friday():
    return render_template('black_friday.html')


@app.route('/black_friday_prediction', methods=['POST'])
def black_friday_prediction():
    if request.method == 'POST':
        form_data = request.get_json()

        city_category_options = np.array([1, 2, 3])
        city_category_one_hot = city_category_options == int(
            form_data['city_category'])
        city_category_one_hot = np.array(
            list(map(int, city_category_one_hot))).astype('float')

        # prepare data array
        input_data = np.array([
            form_data['gender'],
            form_data['age'],
            form_data['occupation'],
            form_data['stay_in_current_city'],
            form_data['marital_status'],
            form_data['product_main_category'],
            form_data['product_category_1'],
            form_data['product_category_2'],
        ]).astype('float')

        input_data_ = np.append(input_data, city_category_one_hot)
        input_data_ = input_data_.reshape(-1, 1)

        # scaling data using standardScaler
        std_scaler = StandardScaler()
        input_data_scalled = std_scaler.fit_transform(input_data_)

        # predict now
        input_data_final = input_data_scalled.reshape(1, -1)
        print(input_data_final)
        pred = black_friday_dt_model.predict(input_data_final)

        return render_template('model_result.html', message={"type": 'normal', 'text': f"Predicted purchase amount for above customer: <b>${str(round(pred[0], 2))}</b>"})


@app.route('/banknotes_authentication')
def banknotes_authentication():
    return render_template('banknotes_authentication.html')


@app.route('/banknotes_auth', methods=['POST'])
def banknotes_auth():
    if request.method == 'POST':
        form_data_json = request.get_json()

        try:
            model_input = np.array([
                form_data_json['variance'],
                form_data_json['skewness'],
                form_data_json['curtosis'],
                form_data_json['entropy']]).astype('float')

            model_input = model_input.reshape(1, - 1)
            predict = banknote_model.predict(model_input)
        except:
            return render_template('model_result.html', message={'type': 'error', 'text': f"Something went wrong!"})

        output_str = ["invalid", "valid"]

        return render_template('model_result.html', message={'type': 'normal', 'text': f"Model says it's a <b>{output_str[predict[0]]}</b> banknote."})


@app.route('/sign_language_recognition_resnet')
def sign_language_recognition_resnet():
    return render_template('resnet.html')


@app.route('/alpaca_mobilenetv2')
def alpaca_mobilenetv2():
    return render_template('alpaca_mobilenetv2.html')

@app.route('/recommender_content_based', methods=['GET', 'POST'])
def recommender_content_based():
    if request.method == 'GET':
        return render_template('recommender_content_based.html', options=recommender_system_df.index)
    elif request.method == 'POST':
        input_movie = request.get_json()
        input_movie_title = input_movie['input_movie']
        
        output_message_type = "normal"
        grid_info = [{
                    'title': input_movie_title,
                    'year': recommender_system_df.loc[input_movie_title]['Year'],
                    'director': recommender_system_df_orig.loc[input_movie_title, 'Director'],
                    'poster': recommender_system_df.loc[input_movie_title]['Poster'],
                    'imdb': recommender_system_df.loc[input_movie_title]['imdbRating']
                }]
        
        recommended_movies = recommender(input_movie_title)
                
        #print(recommended_movies)
        if len(recommended_movies) == 0:
            recommended_movies = "No result found"
            output_message_type = "info"
        else:            
            for movie in recommended_movies:
                grid_info.append({
                    'title': movie,
                    'year': recommender_system_df.loc[movie]['Year'],
                    'director': recommender_system_df_orig.loc[movie, 'Director'],
                    'poster': recommender_system_df.loc[movie]['Poster'],
                    'imdb': recommender_system_df.loc[movie]['imdbRating']
                })
        
        return render_template('info_grid.html', message={'type': output_message_type, 'text': recommended_movies, 'grid_info': grid_info})

@app.route('/co2_emission_lstm', methods=['GET', 'POST'])
def co2_emission_lstm():
    if request.method == 'GET':
        return render_template('co2_emission_lstm.html', params={'last_year': int(co2_orig_csv['year'].iloc[-1])})
    elif request.method == 'POST':
        request_json = request.get_json()
        
        window_size = 2
        
        # validation set
        val_data = np.array([[0.42428668],
                            [0.47168195],
                            [0.49305629],
                            [0.5446335 ],
                            [0.47121728],
                            [0.5265118 ],
                            [0.49026836],
                            [0.46610604],
                            [0.50234947],
                            [0.61154451],
                            [0.67799085],
                            [0.56786648],
                            [0.71284033],
                            [0.8410864 ],
                            [0.91403795],
                            [0.98048429],
                            [0.90288613],
                            [1.        ]])
        
        x_input = val_data[-(window_size):].reshape(1, -1)
        
        temp_input = list(x_input)
        temp_input = temp_input[0].tolist() # this is the inputs for starting prediction
        
        lst_output=[]
        n_steps=window_size
        number_of_years_to_predict = int(request_json['to_year']) - 2019
        i=0
        while(i<number_of_years_to_predict):
            
            if(len(temp_input)>n_steps):
                #print(temp_input)
                x_input=np.array(temp_input[1:])
                #print("{} Year input {}".format(i,x_input))
                x_input=x_input.reshape(1,-1)
                x_input = x_input.reshape((1, n_steps, 1))
                #print(x_input)
                yhat = co2_emission_lstm_model.predict(x_input, verbose=0)
                #print("{} Year output {}".format(i,yhat))
                temp_input.extend(yhat[0].tolist())
                temp_input=temp_input[1:]
                #print(temp_input)
                lst_output.extend(yhat.tolist())
                i=i+1
            else:
                x_input = x_input.reshape((1, n_steps,1))
                yhat = co2_emission_lstm_model.predict(x_input, verbose=0)
                #print(yhat[0])
                temp_input.extend(yhat[0].tolist())
                #print(len(temp_input))
                lst_output.extend(yhat.tolist())
                i=i+1
        
        final_output = co2_emission_scaler.inverse_transform(lst_output).flatten()
        x_labels = co2_orig_csv['year'].tolist() + np.arange(int(co2_orig_csv['year'].iloc[-1]) + 1, int(request_json['to_year']) + 1).tolist()
        
        chart_data_orig = list(co2_orig_csv['value']) + [0] * len(final_output)
        chart_data_predicted = [0] * (len(co2_orig_csv['value']) - 1) + list([co2_orig_csv['value'].iloc[-1]]) + list(final_output)
        
        response_json = {"x_labels": x_labels, "orig": chart_data_orig, "predicted": chart_data_predicted}
        return jsonify(response_json)

if __name__ == "__main__":
    # app.run(host='0.0.0.0')
    app.run(debug=True)
