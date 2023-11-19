from dash import Dash, html, dcc, callback, Output, Input, ctx, State
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
import csv
import sqlite3
import sqlalchemy
import datetime

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
    'lineColor': '#687879',
    'gridColor': '#384d4f'
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


app = Dash(external_stylesheets=[dbc.themes.SOLAR])  # Dash(__name__)


@app.callback(
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


@app.callback(
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
    print('len df 2: \n', filtered_df)
    fig = updateChartLayout(filtered_df, fig, chart_title, pieChartHeight)

    return fig


@app.callback(
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


@app.callback(
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
                         )

        chart_title = "Landing outcome of Launch Site %s against Payload Mass" % (
            launch_site)

    fig.update_yaxes(zeroline=False,
                     tickvals=[0, 1], linecolor=colors['lineColor'], gridcolor=colors['gridColor'])
    fig.update_xaxes(zeroline=False, linecolor=colors['lineColor'],
                     gridcolor=colors['gridColor'])

    fig = updateChartLayout(filtered_df, fig, chart_title, 250)

    fig.update_layout(
        paper_bgcolor='rgba(0, 0, 0, 0.2)',
    )

    return fig


# YEARLY TREND CHARTS


@app.callback(
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

    filtered_df = filtered_df[['Year', 'Class']
                              ].groupby('Year').mean().reset_index()
    # print(filtered_df)
    fig = px.line(
        filtered_df,
        x='Year',
        y='Class',
        color_discrete_sequence=['#50C878'],
        labels={
            'Class': 'Landing Outcome'
        }
    )

    fig.update_yaxes(
        zeroline=False, linecolor=colors['lineColor'], gridcolor=colors['gridColor'])
    fig.update_xaxes(zeroline=False, linecolor=colors['lineColor'], type='category',
                     gridcolor=colors['gridColor'])

    fig = updateChartLayout(filtered_df, fig, chart_title, 300)

    fig.update_layout(
        paper_bgcolor='rgba(0, 0, 0, 0.2)',
    )

    return fig


@app.callback(
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

    filtered_df = filtered_df[['Year']].value_counts().reset_index()

    fig = px.bar(
        filtered_df,
        x='Year',
        y='count',
        text_auto='.2s',
        color_discrete_sequence=[px.colors.qualitative.Prism[3]],
        labels={
            'count': 'Number of Launches'
        }
    )

    fig.update_yaxes(
        zeroline=False, linecolor=colors['lineColor'], gridcolor=colors['gridColor'])
    fig.update_xaxes(zeroline=False, linecolor=colors['lineColor'], type='category',
                     gridcolor=colors['gridColor'], categoryorder='category ascending')

    fig = updateChartLayout(filtered_df, fig, chart_title, 300)
    fig.update_layout(
        paper_bgcolor='rgba(0, 0, 0, 0.2)',
    )

    return fig


@app.callback(
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


falcon9_layout = html.Div(
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
                                                src=app.get_asset_url(
                                                    'Falcon_9_logo.png'),
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
                                                    for yr in list(range(df['Year'].min(), df['Year'].max() + 1))
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

if __name__ == '__main__':
    app.run(debug=True)
