import dash
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
import numpy as np
import dash_bootstrap_components as dbc
from plotly.graph_objs import Scatter

# Load your DataFrame here
df = pd.read_csv('movingdf3.csv')
# Load additional variables
av = pd.read_csv('additional_vars.csv', sep=";", parse_dates=['date'], nrows=4)

for col in av.columns[1:4]:
    av[col] = av[col].astype(float)
# Assuming 'relevant_columns' contains the list of columns you want to include
relevant_columns = av.columns[1:4].tolist()  # Or specify your columns ['col1', 'col2', ...]

additional_vars_options = [{'label': var, 'value': var} for var in relevant_columns]


welle_to_date = {
    -1: '2024-01-01',
    0: '2023-01-01',
    1: '2022-06-01',
    2: '2022-01-01'
}

# Apply the mapping to create a new 'date' column
df['date'] = df['Welle'].map(welle_to_date)

# Identify relevant columns for analysis
f_classes = [
    "F35A1", "F35A2", "F35A3", "F35A4", "F35A5",
    "F36A2", "F36A3", "F36A4", "F36A6", "F36A7", "F36A8", "F36A10", "F36A11", "F36A13", "F36A14"
]

# Mapping codes to descriptions
code_to_description = {
    "F35A1": "Klasse A1",
    "F35A2": "Klasse A2",
    "F35A3": "Klasse AM",
    "F35A4": "Klasse B",
    "F35A5": "Klasse CE – inkl. C + Zusatzqualifikation",
    "F36A2": "Praktische Stunde A1 45 Minuten",
    "F36A3": "Praktische Stunde A2 45 Minuten",
    "F36A4": "Praktische Stunde AM 45 Minuten",
    "F36A6": "Praktische PKW-Stunde 45 Minuten",
    "F36A7": "Theoriestunde PKW Online 45 Minuten",
    "F36A8": "Theoriestunde PKW in Präsenz 45 Minuten",
    "F36A10": "Praktische LKW-Stunde Klasse C 45 Minuten",
    "F36A11": "Praktische LKW-Stunde Klasse CE 45 Minuten",
    "F36A13": "Praktische Bus-Stunde Klasse D 45 Minuten",
    "F36A14": "Praktische Bus-Stunde Klasse DE 45 Minuten"
}


umsatzklassen_labels_to_values = {
    "Weniger als 100.000 €": 1,
    "100.000 € bis unter 250.000 €": 2,
    "250.000 € bis unter 500.000 €": 3,
    "500.000 € bis unter 1.000.000 €": 4,
    "1.000.000 € und mehr": 5,
    "Keine Angabe": 6,
}

wohnumfeld_labels_to_values = {
    "Ländliche Gegend": 1 ,
    "Kleinstadt / Mittelgroße Stadt": 2,
    "Großstadt": 3,
}

region_labels_to_values = {
    "West": 1,
    "Ost": 2,
}


# Dummy data for dropdowns, replace with actual data from your DataFrame
# Convert dictionary labels to dropdown options
umsatzklassen_options = [{'label': label, 'value': value} for label, value in umsatzklassen_labels_to_values.items()]
wohngegend_options = [{'label': label, 'value': value} for label, value in wohnumfeld_labels_to_values.items()]
region_options = [{'label': label, 'value': value} for label, value in region_labels_to_values.items()]

umsatzklassen_default_values = [option['value'] for option in umsatzklassen_options]
region_default_values = [option['value'] for option in region_options]
wohngegend_default_values = [option['value'] for option in wohngegend_options]

f_class_default_values = f_classes


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.Img(src='/assets/0_moving.jpg', style={'width': '100%', 'height': 'auto'}), width=12)
    ]),
    dbc.Row([
        dbc.Col(dbc.Card(id="kpi-1", body=True, color="light", style={"borderRadius": "15px", "margin": "10px"}), width=4),
        dbc.Col(dbc.Card(id="kpi-2", body=True, color="light", style={"borderRadius": "15px", "margin": "10px"}), width=4),
        dbc.Col(dbc.Card(id="kpi-3", body=True, color="light", style={"borderRadius": "15px", "margin": "10px"}), width=4)
    ], style={"marginTop": "20px"}),

    dbc.Row([
        dbc.Col(html.Div([
            html.H6("Umsatzklassen"),
            dcc.Dropdown(
                id='umsatzklassen-dropdown',
                options=umsatzklassen_options,
                value=umsatzklassen_default_values,  # Set all options selected by default
                multi=True,
                placeholder='Select Umsatzklassen'
            )
        ]), width=4),
        dbc.Col(html.Div([
            html.H6("Region"),
            dcc.Dropdown(
                id='region-dropdown',
                options=region_options,
                value=region_default_values,  # Set all options selected by default
                multi=True,
                placeholder='Select Region'
            )
        ]), width=4),
        dbc.Col(html.Div([
            html.H6("Wohngegend"),
            dcc.Dropdown(
                id='wohngegend-dropdown',
                options=wohngegend_options,
                value=wohngegend_default_values,  # Set all options selected by default
                multi=True,
                placeholder='Select Wohngegend'
            )
        ]), width=4),
    ]),
    dbc.Row([
        dbc.Col(html.Div([
            html.H6("Führerscheinklassen", style={'margin-top': '20px'}),
            dcc.Dropdown(
                id='f-class-dropdown',
                options=[{'label': code_to_description.get(f, f), 'value': f} for f in f_classes],
                multi=True,
                placeholder='Select Führerscheinklassen',
                value=[f_classes[0]] if f_classes else []  # Default selection of the first class
            )
        ]), width=12),
    ]),

    dbc.Row([
    dbc.Col(html.Div([
        html.H6("Zusätzliche Variablen", style={'margin-top': '20px'}),
        dcc.Dropdown(
            id='additional-vars-dropdown',
            options=additional_vars_options,
            multi=True,
            placeholder='Zusätzliche Variablen auswählen',
        )
    ]), width=12),
]),

   dbc.Row([
    dbc.Col(html.Div(id='additional-vars-values-display'), width=12),
]),




    dbc.Row([
        dbc.Col(html.H2("Preis- / Gehaltsentwicklungen am Fahrschulmarkt", style={'text-align': 'center', 'margin-top': '50px'}), width=12)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='price-change-line-chart'), width=12),
    ]),
    dbc.Row([
        dbc.Col(html.Div("Quelle: MOVING Fahrschul-Monitor", style={'text-align': 'left', 'margin-top': '20px'}), width=12)
    ])
], fluid=True, style={"maxWidth": "1200px"})


@app.callback(
    Output('price-change-line-chart', 'figure'),
    [Input('umsatzklassen-dropdown', 'value'),
     Input('region-dropdown', 'value'),
     Input('wohngegend-dropdown', 'value'),
     Input('f-class-dropdown', 'value'),
     Input('additional-vars-dropdown', 'value')]  # New input for additional variables
)
def update_graph(umsatzklassen, region, wohngegend, selected_classes, selected_additional_vars):
    # Filter based on dropdown selections
    umsatzklassen = [umsatzklassen] if not isinstance(umsatzklassen, list) else umsatzklassen
    region = [region] if not isinstance(region, list) else region
    wohngegend = [wohngegend] if not isinstance(wohngegend, list) else wohngegend

    # Merge the filtered_data with the additional variables dataframe (av) on the 'date' column
    filtered_data = df[df['S4'].isin(umsatzklassen) & df['S5f'].isin(region) & df['S6.1'].isin(wohngegend)]
    filtered_data['date'] = pd.to_datetime(filtered_data['Welle'].map(welle_to_date))
    filtered_data = filtered_data.replace(-50, np.nan)

    # Calculate the average prices and percenstage differences for the selected classes
    average_prices = filtered_data.groupby('date')[selected_classes].mean()
    percentage_differences = average_prices.pct_change().fillna(0) * 100 + 100

    # Create traces for the selected classes
    traces = []
    for f_class in selected_classes:
        traces.append(go.Scatter(
            x=percentage_differences.index,
            y=percentage_differences[f_class],
            mode='lines+markers',
            name=code_to_description.get(f_class, f_class),
            line_shape='spline',
            line=dict(smoothing=1.3, width=3)
        ))

    # Add traces for selected additional variables directly from the av DataFrame
    if selected_additional_vars:
        for var in selected_additional_vars:
            # Ensure that the 'date' columns in both DataFrames are in the same format and comparable
            av['date'] = pd.to_datetime(av['date'])
            traces.append(go.Scatter(
                x=av['date'],
                y=av[var],
                mode='lines+markers',
                name=var,
                line_shape='spline',
                line=dict(smoothing=1.3, width=3)
            ))

    # Return the figure object with data and layout
    return {
        'data': traces,
        'layout': go.Layout(
            title='Prozentuale Veränderung (YoY) vs. Inflationsrate',
            yaxis={'title': 'Änderungsrate in %', 'range': [80, 150]},
            margin={'l': 40, 'b': 40, 't': 40, 'r': 10},
            legend={'x': 0, 'y': 1},
            hovermode='closest'
        )
    }



@app.callback(
    [Output('kpi-1', 'children'),
     Output('kpi-2', 'children'),
     Output('kpi-3', 'children')],
    [Input('f-class-dropdown', 'value')]
)


def update_kpi_tiles(selected_classes):
    if not selected_classes or len(selected_classes) < 3:
        # Provide default content if fewer than 3 classes are selected
        default_content = "Bitte wählen Sie drei Variablen aus."
        return [html.Div(default_content)] * 3

    # Handling special cases for Welle values
    # Assuming -1 is the most recent and 0 is the second most recent
    latest_welle = -1  # Directly set to -1, as it's the most recent by definition
    previous_welle = 0  # Directly set to 0, as it's the second most recent by definition

    # Replace -50 with np.nan to avoid affecting calculations
    df_copy = df.replace(-50, np.nan)

    # Filter the DataFrame for the latest and previous Welle values
    latest_data = df_copy[df_copy['Welle'] == latest_welle]
    previous_data = df_copy[df_copy['Welle'] == previous_welle]

    # Calculate the latest and previous average prices for the selected classes
    latest_avg_prices = latest_data[selected_classes[:3]].mean()
    previous_avg_prices = previous_data[selected_classes[:3]].mean()

    # Calculate percentage changes
    percent_changes = ((latest_avg_prices - previous_avg_prices) / previous_avg_prices) * 100

    # Prepare the content for each KPI tile
    kpi_contents = []
    for i, f_class in enumerate(selected_classes[:3]):
        class_description = code_to_description.get(f_class, f_class)
        avg_price = latest_avg_prices.iloc[i]
        percent_change = percent_changes.iloc[i]

        # Format the content string with average price and percent change
        content_str = f"{class_description}: {avg_price:.2f} € ({percent_change:+.2f}%)"
        content_div = html.Div(content_str, style={"padding": "10px"})  # Style as needed

        kpi_contents.append(content_div)

    # Fill in remaining tiles if fewer than 3 classes are selected
    while len(kpi_contents) < 3:
        kpi_contents.append(html.Div("N/A"))

    return kpi_contents


@app.callback(
    Output('additional-vars-values-display', 'children'),
    [Input('additional-vars-dropdown', 'value')]
)
def update_additional_vars_display(selected_additional_vars):
    if not selected_additional_vars:
        return "Bitte wählen Sie zusätzliche Variablen aus und sie werden hier angezeigt."

    # Create a dataframe with the dates and selected variables
    av_selected = av[['date'] + selected_additional_vars]
    # Convert the dataframe to a HTML table
    return dbc.Table.from_dataframe(av_selected, striped=True, bordered=True, hover=True)



if __name__ == '__main__':
    app.run_server(debug=True)
