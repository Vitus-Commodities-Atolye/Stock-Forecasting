import datetime

import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go

# reading the csv files and convert them to data frame
stocks = pd.read_csv("historical_results.csv")
stocks['Date'] = pd.to_datetime(stocks['Date'])

tickers = stocks['Stock'].sort_values().unique()
min_date = stocks['Date'].min()
max_date = (stocks['Date'].max() - pd.DateOffset(days=30)).date()

forecasts = pd.read_csv("prediction_results.csv")
forecasts['Date'] = pd.to_datetime(forecasts['Date'])

# init Dash app
app = Dash(__name__)
app.title = "Stock Market Insider"

# layout of the Dash app
app.layout = html.Div([
    html.Header([
        html.H1("Stock Market"),
        html.H1("Insider")
    ], className="header"),
    html.Main([
        html.Div(
            [
                html.Div(
                    [
                        html.P("Select stock:"),
                        dcc.Dropdown(options=tickers, value=None, id="stock_code")
                    ], className="input"
                ),
                html.Div(
                    [
                        html.P("Compare stock:"),
                        dcc.Dropdown(options=tickers, value=None, id="stock_code_cmp")
                    ], className="input"
                ),
                html.Div(
                    [
                        html.P("Select interval:"),
                        # Date range input, default value will be defined from data
                        dcc.DatePickerRange(
                            id='date-range',
                            min_date_allowed=min_date,
                            max_date_allowed=max_date,
                            start_date=min_date,
                            end_date=max_date
                        )
                    ], className="date"
                ),
                html.Div(
                    [
                        html.P("Forecast range in days:"),
                        dcc.Dropdown(options=[7, 14, 30], value=7,
                                     id="forecast_day")
                    ], className="input"
                ),

            ], className="input-box"),

        html.Div([
            html.Div(
                [
                    dcc.Loading(children=[html.Div([], id='stonks-graph', className='graphs')], id='loading1',
                                type='graph'),
                    dcc.Loading(id='loading2',
                                children=[html.Div([], id='forecast-graph', className='graphs')],
                                type='graph')
                ], className="graphs"
            )

        ], className='outputContainer'
        )
    ], className='container'
    ),
    html.Footer([
        html.Div(
            html.P("Copyright © 2023"),
            className="copyright"
        )
    ])
])

"""
Precondition : Ticker name and start date should be defined. To compare with another ticker needs a ticker name
should be provided (optional).

Post-condition : The stock trend of the ticker with given date is provided.

@:param stock_code : ticker name
@:param stock_code_cmp (optional) : ticker name for comparing the stock trend of the given ticker
@:param date_range : start date of the ticker's stock trend
@:return stonks-graph : graphs of the stock trend
"""


@app.callback(
    Output('stonks-graph', 'children'),
    Input('stock_code', 'value'),
    Input('stock_code_cmp', 'value'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date'),
    Input('forecast_day', 'value')
)
def update_stock_graph(stock_code, stock_code_cmp, start_date, end_date, interval):
    # Create the figure and add two scatter traces
    fig = go.Figure()

    # query for gathering related ticker data.
    stock = stocks.query(f"Stock in ['{stock_code}']")
    if not stock.empty:
        p_end_date = (pd.Timestamp(end_date) + pd.DateOffset(days=int(interval))).date()

        stock_interval = stock.loc[
            (stock['Date'] >= pd.Timestamp(start_date)) & (stock['Date'] <= pd.Timestamp(end_date))]
        # draws a line of the given ticker's data
        fig.add_trace(go.Scatter(
            x=stock_interval['Date'],
            y=stock['Close'],
            mode='lines',
            name=f'{stock_code}',
        ))

        forecast = forecasts.query(f"Stock in ['{stock_code}']")
        forecast_interval = forecast.loc[
            (forecast['Date'] >= (pd.Timestamp(end_date) - pd.DateOffset(days=2))) & (forecast['Date'] <= pd.Timestamp(p_end_date))]
        print(forecast_interval)
        fig.add_trace(go.Scatter(
            x=forecast_interval['Date'],
            y=forecast['Prediction'],
            mode='lines',
            name=f'{stock_code} Prediction',
            line=dict(dash='dash'),
        ))

        fig.add_trace(go.Scatter(
            x=forecast_interval['Date'],
            y=forecast_interval['Actual'],
            mode='lines',
            name=f'{stock_code} Actual',
        ))

    return dcc.Graph(figure=fig)


if __name__ == '__main__':
    app.run_server(debug=True)
