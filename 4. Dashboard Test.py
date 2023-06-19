# HELP AND CHEATSHEETS
#-------------------------------------------------------------------
#https://hackerthemes.com/bootstrap-cheatsheet/#mt-1
#Bootstrap Themes: https://bootswatch.com/flatly/

# IMPORT LIBRARIES
#-------------------------------------------------------------------
import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np
from dash import Dash, dcc, html, Input, Output, callback
from dash import Dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib as mpl
import locale
import matplotlib.ticker as mticker

# IMPORT DATA
#-------------------------------------------------------------------
#Verkehrsdaten
folder_path = 'C:/Users/michi/OneDrive/MSC DV/Python/Projektarbeit DV/flow'  #Ordnerpfad allenfalls anpassen
data_frames = []

for file in os.listdir(folder_path):
    if file.endswith('.csv'):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)
        filename = os.path.splitext(file)[0]  #Dateiname
        data['Messstation'] = filename[:3]  #Erste drei Zeichen aus Dateinamen für Messstation
        data['Spur'] = filename[5:6]  #Zeichen 5-6 aus Dateinamen für Spur
        data_frames.append(data)

data_flow = pd.concat(data_frames, ignore_index=True) #enthält die aggregierten Daten mit der zusätzlichen Information "Messtation" in der letzten Spalte

#Spalten benennen
data_flow = data_flow.drop("id", axis="columns")
data_flow.columns = data_flow.columns.str.replace('index', 'Fahrzeugkategorie')
data_flow.columns = data_flow.columns.str.replace('measurement_time', 'Zeitstempel')
data_flow.columns = data_flow.columns.str.replace('measured_value', 'Anzahl Fahrzeuge')

#Filter und Datenbereinigung
data_flow = data_flow[data_flow['Fahrzeugkategorie'] == 11] #nur PKW benötigt
data_flow = data_flow.drop(data_flow[data_flow['Anzahl Fahrzeuge'] == 0].index) #lösche alle Zeilen, in denen der Wert 0 hat --> werden nicht benötigt
schwellenwerte = {
    '025': 600,
    '132': 400,
    '228': 550,
    '245': 600,
    '318': 550,
    '373': 400,
    '603': 500,
    '708': 400,
    '828': 550
} #Ausreisser werden so behandelt, dass der Wert davor genommen wird. Annahme, dass sich je Viertel Stunde die Anzahl der Fahrzeuge nicht massiv ändert. Zudem sind nur eine Handvoll Ausreisser vorhanden.
for index, row in data_flow.iterrows():
    messstation = row['Messstation']
    schwellenwert = schwellenwerte.get(messstation)
    if schwellenwert is not None and row['Anzahl Fahrzeuge'] > schwellenwert:
        data_flow.at[index, 'Anzahl Fahrzeuge'] = data_flow.at[index-1, 'Anzahl Fahrzeuge']

#neue Spalten hinzufügen & entfernen
data_flow['Datum'] = pd.to_datetime(data_flow['Zeitstempel'].str[:10], format='%Y-%m-%d') #, errors='coerce').dt.date wieso kommt immer Fehler?
data_flow['Jahreszeit'] = data_flow['Datum'].dt.month.map({1: 'Winter', 2: 'Winter', 3: 'Winter', 4: 'Winter', 5: 'Sommer', 6: 'Sommer', 7: 'Sommer', 8: 'Sommer', 9: 'Sommer', 10: 'Sommer', 11: 'Winter', 12: 'Winter'})
data_flow['Wochentag'] = data_flow['Datum'].dt.day_name()
data_flow['Wochenende'] = np.where(data_flow['Wochentag'].isin(['Saturday', 'Sunday']), 'ja', 'nein')
data_flow = data_flow.drop('Zeitstempel', axis=1)
data_flow = data_flow[data_flow['Wochenende'] == 'ja']

#Sonnenstunden
data_sun_relativ = pd.read_table('Wetterdaten/Sonnenscheindauer_relativ.txt', delimiter=";")
del data_sun_relativ["stn"]
data_sun_relativ.columns = data_sun_relativ.columns.str.replace('time', 'Datum')
data_sun_relativ.columns = data_sun_relativ.columns.str.replace('sremaxdv', 'Sonnenstunden relativ')
data_sun_relativ['Datum'] = pd.to_datetime(data_sun_relativ['Datum'], format='%Y%m%d')

#Aggregation
df = pd.merge(data_flow, data_sun_relativ, on=['Datum'], how='inner')

#alle Koordinaten der Messstationen ergänzen
condition_025 = df['Messstation'] == '025'
condition_132 = df['Messstation'] == '132'
condition_228 = df['Messstation'] == '228'
condition_245 = df['Messstation'] == '245'
condition_318 = df['Messstation'] == '318'
condition_373 = df['Messstation'] == '373'
condition_603 = df['Messstation'] == '603'
condition_708 = df['Messstation'] == '708'
condition_828 = df['Messstation'] == '828'


df['Längengrad'] = np.where(condition_025, 9.506224, np.where(condition_228, 9.562032, np.where(condition_318, 9.431858, np.where(condition_373, 9.834898, np.where(condition_828, 9.500202, np.where(condition_132, 9.604596, np.where(condition_708, 9.760102, np.where(condition_245, 9.440388, np.where(condition_603, 9.334899, np.nan)))))))))
df['Breitengrad'] = np.where(condition_025, 47.014340, np.where(condition_228, 46.927032, np.where(condition_318, 47.047050, np.where(condition_373, 46.90024, np.where(condition_828, 47.080658, np.where(condition_132, 46.972653, np.where(condition_708, 46.915761, np.where(condition_245, 47.040411, np.where(condition_603, 47.114369, np.nan)))))))))


# FUNKTIONEN
#-------------------------------------------------------------------
def scatter_plot(filtered_df):
    df_scatter = filtered_df.groupby(['Messstation', 'Datum']).agg({'Anzahl Fahrzeuge': 'sum', 'Sonnenstunden relativ': 'first'}).reset_index()
    scatter = px.scatter(df_scatter, x='Anzahl Fahrzeuge',
                         y='Sonnenstunden relativ',
                         color='Messstation',
                         color_discrete_sequence= ['#015666', '#1a889d', '#4da3b3', '#80bdc9', '#b3d7de', '#cce5e9',  '#2b6b51', '#317a5c','#378a68','#50a381', '#77b89d', '#9eccb9'],
                         hover_data=['Datum'])
    scatter.update_layout(xaxis_title="Anzahl Fahrzeuge",
                          yaxis_title="Sonnenstunden relativ",
                          plot_bgcolor='white',
                          paper_bgcolor='white',
                          font_color='black',
                          xaxis=dict(gridcolor='lightgray',zerolinecolor='black'),
                          yaxis=dict(gridcolor='lightgray', zerolinecolor='black'),
                          legend=dict(orientation="h", x=0, y=1.1))
    return scatter

def get_total_values(filtered_df):
    total_values = filtered_df.groupby('Messstation')['Anzahl Fahrzeuge'].sum().reset_index()
    return total_values

def balken_linien_diagramm(filtered_df):
    df_messstation_davos = df[df['Messstation'] == '373']
    df_messstation_davos = df_messstation_davos.groupby('Datum').agg({'Anzahl Fahrzeuge': 'sum'}).reset_index()
    df_balken_linien = filtered_df.groupby('Datum').agg({'Anzahl Fahrzeuge': 'sum'}).reset_index()
    df_balken_linien = df_balken_linien.sort_values('Datum')
    df_balken_linien = pd.merge(df_balken_linien, df_messstation_davos, on='Datum', suffixes=('', '_davos'))
    df_balken_linien['Anteil Davos (%)'] = df_balken_linien['Anzahl Fahrzeuge_davos'] / df_balken_linien['Anzahl Fahrzeuge'] *100

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_balken_linien['Datum'],
        y=df_balken_linien['Anzahl Fahrzeuge'],
        name='Anzahl Fahrzeuge',
        marker_color='#80bdc9',
        yaxis='y',
    ))

    fig.add_trace(go.Scatter(
        x=df_balken_linien['Datum'],
        y=df_balken_linien['Anteil Davos (%)'],
        name='Anteil Davos (%)',
        mode='lines',
        line_color='#015666',
        yaxis='y2',
    ))

    fig.update_layout(
        xaxis_title='Datum',
        #xaxis=dict(type='category', categoryorder='array', categoryarray=np.unique(df_balken_linien['Datum']), gridcolor='lightgray', zerolinecolor='black'),
        xaxis=dict(
            type='date',
            tickformat='%Y-%m-%d',
            tickangle=-45,
            gridcolor='lightgray',
            zerolinecolor='black'),
        yaxis=dict(title='Anzahl Fahrzeuge', color='#80bdc9', gridcolor='lightgray', zerolinecolor='black'),
        yaxis2=dict(title='Anteil Davos (%)', color='#015666', gridcolor='rgba(0,0,0,0)', zerolinecolor='black', overlaying='y', side='right', rangemode='tozero'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color='black',
        barmode='stack',
        legend=dict(orientation="h", x=0.3, y=1.1)
    )

    return fig


def grupp_balken_diagramm(filtered_df):
    grouped_balken = filtered_df.groupby(['Datum', 'Messstation']).agg({'Anzahl Fahrzeuge': 'sum'}).reset_index()
    fig = go.Figure()

    colors = ['#015666', '#1a889d', '#4da3b3', '#80bdc9', '#b3d7de', '#cce5e9', '#2b6b51', '#317a5c', '#378a68',
              '#50a381', '#77b89d', '#9eccb9']

    for i, station in enumerate(grouped_balken['Messstation'].unique()):
        data_station = grouped_balken[grouped_balken['Messstation'] == station]
        fig.add_trace(go.Bar(
            x=data_station['Datum'],
            y=data_station['Anzahl Fahrzeuge'],
            name=f'{station}',
            marker_color=colors[i % len(colors)]
        ))

    fig.update_layout(
        xaxis_title='Datum',
        yaxis_title='Anzahl Fahrzeuge',
        barmode='group',
        legend=dict(orientation="h", x=0, y=1.1),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color='black',
        xaxis=dict(gridcolor='lightgray', zerolinecolor='black'),
        yaxis=dict(gridcolor='lightgray', zerolinecolor='black')
    )

    return fig

def gauge_chart_corr(filtered_df):
    correlation = filtered_df['Anzahl Fahrzeuge'].corr(filtered_df['Sonnenstunden relativ'])

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=correlation,
        number={'font': {'size': 30}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': [-1, 1]},
               'bar': {'color': '#015666'},
               'steps': [
                   {'range': [-1, -0.3], 'color': '#80bdc9'},
                   {'range': [-0.3, 0.3], 'color': '#b3d7de'},
                   {'range': [0.3, 1], 'color': '#80bdc9'}
               ]}
    ))

    fig.update_layout(
        title={'text': 'Korr. Fz. / Sonnenstunden', 'x': 0.5, 'y': 0.8,
               'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 14}},
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color='black'
    )

    return fig


def gauge_chart_anz(filtered_df):
    df_messstation_davos = df[df['Messstation'] == '373']
    df_messstation_davos_sum = df_messstation_davos['Anzahl Fahrzeuge'].sum()
    df_other_sum = filtered_df[filtered_df['Messstation'] != '373']['Anzahl Fahrzeuge'].sum()
    proportion = round(df_messstation_davos_sum / (df_messstation_davos_sum + df_other_sum) * 100, 2)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proportion,
        number={'suffix': '%', 'font': {'size': 30}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': '#015666'},
               'steps': [
                   {'range': [0, proportion], 'color': '#015666'},
                   {'range': [proportion, 100], 'color': '#b3d7de'}
               ]}
    ))

    fig.update_layout(
        title={'text': 'Anteil Fahrzeuge Station 373', 'x': 0.5, 'y': 0.8,
               'xanchor': 'center', 'yanchor': 'top'},
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color='black'
    )

    return fig


def map_flow(filtered_df):
    grouped_data = filtered_df.groupby(['Längengrad', 'Breitengrad']).agg({'Anzahl Fahrzeuge': 'sum', 'Messstation': 'first'}).reset_index()

    fig = px.scatter_mapbox(grouped_data, lat='Breitengrad', lon='Längengrad', size='Anzahl Fahrzeuge', hover_data=['Messstation'])
    fig.update_layout(mapbox_style="carto-positron", #oder "open-street-map"?
                      mapbox_center={"lat": 47.0014, "lon": 9.5040},
                      mapbox_zoom=9) #Bad Ragaz als Zentrum
    fig.update_traces(marker=dict(color='#015666'))
    fig.update_layout(height=500, width=800)

    return fig


# START APP
#-------------------------------------------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY],

                # make it mobile-friendly
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )

# LAYOUT SECTION: BOOTSTRAP
#--------------------------------------------------------------------
app.layout = html.Div([
    html.H1("Verkehrsdaten nach Davos / Klosters"),

    dcc.Loading(
        id="loading",
        type="default",
        children=[
            dbc.Row([
                dbc.Col(
                    dbc.Alert(
                        "Messstationen: 025: Bad Ragaz (A13), Station 132: Landquart (A28), Station 228: Zizers (A13), Station 245: Sargans (A3), Station 318: Mels (A3), Station 373: Klosters (A28), Station 603: Walenstadt (A3), Station 708: Kueblis (A28), Station 828: Trübbach (A13)",
                        color='#015666',
                        style={'margin-top': '1px', 'margin-bottom': '10px', 'width': '100%'}
                    ),
                    width=12
                ),

                dbc.Col([
                    dcc.DatePickerRange(
                        id='date-slider',
                        min_date_allowed=df['Datum'].min(),
                        max_date_allowed=df['Datum'].max(),
                        start_date=df['Datum'].min(),
                        end_date=df['Datum'].max(),
                        display_format='YYYY-MM-DD',
                        style={'height': '40px'}
                    )
                ], width=6, style={'height': '60px', 'background-color': '#cce5e9', 'width': '33%'}),

                dbc.Col([
                    dcc.Dropdown(
                        id='station-dropdown',
                        options=[{'label': station, 'value': station} for station in df['Messstation'].unique()],
                        value=[],
                        multi=True,
                        placeholder='Messstation auswählen',
                        style={'height': '40px'}
                    )
                ], width=3, style={'height': '60px', 'background-color': '#cce5e9', 'width': '33%'}),#, 'display': 'flex', 'align-items': 'center', 'justify-content': 'center' wieso funktioniert das nicht?

                dbc.Col([
                    dcc.Dropdown(
                        id='season-dropdown',
                        options=[{'label': season, 'value': season} for season in df['Jahreszeit'].unique()],
                        value=[],
                        multi=True,
                        placeholder='Jahreszeit auswählen',
                        style={'height': '40px'}
                    )
                ], width=3, style={'height': '60px', 'background-color': '#cce5e9', 'width': '33%'}),
            ]),

            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='scatter-plot', style={'height': '500px', 'margin-top': '0px'}),
                ], width={'size': 5}),

                dbc.Col([
                    dcc.Graph(id='balken-linien-diagramm', style={'height': '500px', 'margin-top': '0px'}),
                ], width={'size': 5}),

                dbc.Col([
                    dcc.Graph(id='gauge-chart-anz', style={'height': '300px', 'margin-top': '0px'}),
                ], width={'size': 2}),

            ], style={'width': '100%'}),

            dbc.Row([
                    dbc.Col(dcc.Dropdown(
                        id='dropdown-1',
                        options=[{'label': station, 'value': station} for station in df['Messstation'].unique()],
                        value='025', #Wert vordefiniert - allenfalls noch anpassen
                        clearable=False
                    ), width=4),
                    dbc.Col(dcc.Dropdown(
                        id='dropdown-2',
                        options=[{'label': station, 'value': station} for station in df['Messstation'].unique()],
                        value='373', #Wert vordefiniert - allenfalls noch anpassen
                        clearable=False
                    ), width=4),
                ]),

            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='map-flow', style={'height': '500px', 'margin-top': '0px'}),
                ], width=5),

                dbc.Col([
                    dcc.Graph(id='grupp-balken-diagramm', style={'height': '500px', 'margin-top': '0px'}),
                ], width=5),

                dbc.Col([
                    dcc.Graph(id='gauge-chart-corr', style={'height': '300px', 'margin-top': '0px'}),
                ], width=2)
            ], style={'width': '100%'}),
        ]
    )
])


# Callback-Funktionen
@app.callback(
    Output('scatter-plot', 'figure'),
    Output('balken-linien-diagramm', 'figure'),
    Output('grupp-balken-diagramm', 'figure'),
    Output('gauge-chart-anz', 'figure'),
    Output('gauge-chart-corr', 'figure'),
    Output('map-flow', 'figure'),
    Input('date-slider', 'start_date'),
    Input('date-slider', 'end_date'),
    Input('station-dropdown', 'value'),
    Input('season-dropdown', 'value')
)


def update_figures(start_date, end_date, stations, seasons):
    min_date = pd.to_datetime(start_date)
    max_date = pd.to_datetime(end_date)

    filtered_df = df[(df['Datum'] >= min_date) & (df['Datum'] <= max_date)]

    if stations:
        filtered_df = filtered_df[filtered_df['Messstation'].isin(stations)]

    if seasons:
        filtered_df = filtered_df[filtered_df['Jahreszeit'].isin(seasons)]

    scatter_fig = scatter_plot(filtered_df)
    balken_linien_fig = balken_linien_diagramm(filtered_df)
    #grupp_balken_fig = grupp_balken_diagramm(filtered_df)
    gauge_chart_anz_fig = gauge_chart_anz(filtered_df)
    gauge_chart_corr_fig = gauge_chart_corr(filtered_df)
    map_flow_fig = map_flow(filtered_df)

    return scatter_fig, balken_linien_fig, gauge_chart_anz_fig, gauge_chart_corr_fig, map_flow_fig #grupp_balken_fig,

@app.callback(
    Output('grupp_balken_fig', 'figure'),
    [Input('dropdown-1', 'value'), Input('dropdown-2', 'value')]

def update_grupp_balken_fig(messstation_1, messstation_2):
    filtered_df = df[(df['Messstation'] == messstation_1) | (df['Messstation'] == messstation_2)]
    grupp_balken_fig = grupp_balken_diagramm(filtered_df)

    return grupp_balken_fig


# RUN THE APP
#--------------------------------------------------------------------
if __name__=='__main__':
    app.run_server(debug=False, port=8080)