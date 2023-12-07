from dash import Dash, html, dcc, callback, Output, Input, ctx, State
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
import csv
import sqlite3
import sqlalchemy
import datetime
import sys
sys.path.append('../../')
import folium
import os.path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def os_path(filename):
    return os.path.join(BASE_DIR, filename)

colors = {
    'background': '#111111',
    'text': '#becdce',
    'light': '#e7e7e7',
    'dark': '#515151',
    'lineColor': '#687879',
    'gridColor': '#384d4f',
    'br_gridColor': '#454545'
}

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

# CO2 emission Analysis app
app = Dash(external_stylesheets=[dbc.themes.DARKLY, dbc.icons.BOOTSTRAP], routes_pathname_prefix="/co2_emission_dashboard/")
app.title = "CO2 emission"

co2_all_countries = pd.read_csv(os_path('data/all_countries.csv'))
co2_geo_region_df = pd.read_csv(os_path('data/geo_regions.csv'))
co2_economy_region_df = pd.read_csv(os_path('data/economy_groups.csv'))

geo_json_r = open(os_path('data/world-countries.json'), 'r')
geo_json = geo_json_r.read()

@app.callback(
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

@app.callback(
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

@app.callback(
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

@app.callback(
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

@app.callback(
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

app.layout = html.Div(
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

if __name__ == "__main__":
    app.run(debug=True)