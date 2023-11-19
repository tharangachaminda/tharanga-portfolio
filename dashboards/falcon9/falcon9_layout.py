from dashboards.falcon9.falcon9 import *
from dash import html, dcc
import dash_bootstrap_components as dbc

colors = {
    'background': '#111111',
    'text': '#becdce',
    'light': '#e7e7e7',
    'lineColor': '#687879',
    'gridColor': '#384d4f'
}

binary_class_palette = ['#DE3163', '#50C878']

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
                                                src=falapp.get_asset_url(
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
