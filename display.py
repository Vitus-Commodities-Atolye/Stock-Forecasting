import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

# reading the csv files and convert them to data frame
stocks = pd.read_csv("data.csv")
forecasts = pd.read_csv("prediction_results.csv")

# init Dash app
app = Dash(__name__)

# layout of the Dash app
app.layout = html.Div([
    html.Div(
        [
            html.Div(
                html.H1("Stock Forecast App"),
                className="header"
            ),

            html.Div([
                html.Div(
                    [
                        html.P("Select stock:"),
                        dcc.Dropdown(options=stocks["Stock"].unique(), value="Ticker Name", id="stock_code")
                    ], className="inputs"
                ),

                html.Div(
                    [
                        html.P("Select stock you want to compare:"),
                        dcc.Dropdown(options=stocks["Stock"].unique(), value=None, id="stock_code_cmp")
                    ], className="inputs"
                ),

                html.Div(
                    [
                        html.P("Forecast range in days:"),
                        dcc.Dropdown(options=[7, 14, 30], value=7,
                                     id="forecast_day")
                    ], className="inputs"
                ),

                html.Div(
                    [
                        html.P("Select start date:"),
                        # Date range input, default value will be defined from data
                        dcc.DatePickerSingle(
                            id='date-range',
                            min_date_allowed=stocks['Date'].min(),
                            max_date_allowed=stocks['Date'].max(),
                            date=stocks['Date'].min()
                        )
                    ], className="inputs"
                ),
            ], className="input-box"),

            html.Div(
                html.P("Copyright © 2023"),
                className="copyright"
            )
        ], className='nav'),

    html.Div([
        html.Div(
            [
                dcc.Loading(children=[html.Div([], id='stonks-graph', className='graphs')], id='loading1',
                            type='graph'),
                dcc.Loading(id='loading2',
                            children=[html.Div([], id='forecast-graph', className='graphs')],
                            type='graph')
            ]
        )

    ], className='outputContainer'
    )
], className='container'
)


"""
Precondition : Ticker name and start date should be defined. To compare with another ticker needs a ticker name 
should be provided (optional). 

Postcondition : The stock trend of the ticker with given date is provided.

@:param stock_code : ticker name
@:param stock_code_cmp (optional) : ticker name for comparing the stock trend of the given ticker
@:param date_range : start date of the ticker's stock trend
@:return stonks-graph : graphs of the stock trend
"""
@app.callback(
    Output('stonks-graph', 'children'),
    Input('stock_code', 'value'),
    Input('stock_code_cmp', 'value'),
    Input('date-range', 'date')
)
def update_stock_graph(stock_code, stock_code_cmp, date):
    # query for gathering related ticker data.
    df1 = stocks.query(f"Stock in ['{stock_code}']")

    # define start and end dates for date range
    start_date = date;
    end_date = df1['Date'].max()

    # draws a line of the given ticker's data
    fig = px.line(
        df1,
        x='Date',
        y='Close',
        range_x=[start_date, end_date],
        color=df1['Stock']
    )

    if stock_code_cmp is not None:
        # query for gathering related ticker data.
        df2 = stocks.query(f"Stock in ['{stock_code_cmp}']")

        end_date = df2['Date'].max()

        fig.add_trace(px.line(
            df2,
            x='Date',
            y='Close',
            range_x=[start_date, end_date],
            color=df2['Stock'],
            color_discrete_sequence=["red"]
        ).data[0], )

    fig.update_layout(
        title=dict(text="Stock Trend", font=dict(size=30))
    )

    return dcc.Graph(figure=fig)


"""
Precondition : Ticker name and interval should be defined. To compare with another ticker needs a ticker name 
should be provided (optional). 

Postcondition : The forecast trend of the ticker with given interval is provided.

@:param stock_code : ticker name
@:param stock_code_cmp (optional) : ticker name for comparing the stock trend of the given ticker
@:param date_range : interval of the forecast in days
@:return stonks-graph : graphs of the forecasted ticker's trend
"""
@app.callback(
    Output('forecast-graph', 'children'),
    Input('stock_code', 'value'),
    Input('stock_code_cmp', 'value'),
    Input('forecast_day', 'value')
)
def update_forecast_graph(stock_code, stock_code_cmp, interval):
    # query for gathering related ticker data.
    df1 = forecasts.query(f"Stock in ['{stock_code}']")

    # convert 'Date' column to datetime format
    forecasts['Date'] = pd.to_datetime(forecasts['Date'])

    # define start and end dates for date range
    start_date = forecasts['Date'].min().date()
    end_date = (start_date + pd.DateOffset(days=int(interval))).date()

    # draws a line of the given ticker's data
    fig = px.line(
        df1,
        x='Date',
        y='Prediction',
        range_x=[start_date, end_date],
        color=df1['Stock'],
    )

    if stock_code_cmp is not None:
        # query for gathering related ticker data.
        df2 = forecasts.query(f"Stock in ['{stock_code_cmp}']")

        fig.add_trace(px.line(
            df2,
            x='Date',
            y='Prediction',
            range_x=[start_date, end_date],
            color=df2['Stock'],
            color_discrete_sequence=["red"]
        ).data[0], )

    fig.update_layout(
        title=dict(text="Forecast Trend", font=dict(size=30))
    )
    return dcc.Graph(figure=fig)


if __name__ == '__main__':
    app.run_server(debug=True)
