from pyexpat import model
import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
import plotly.graph_objs as go
from datetime import datetime
import holidays

def fetch_stock_data(stock_symbol):
    # Fetch historical stock data
    stock_data = yf.download(stock_symbol, start="2016-01-01", end=datetime.today().strftime('%Y-%m-%d'))
    stock_data.reset_index(inplace=True)
    return stock_data

def add_holidays(df):
    india_holidays = holidays.India(years=df['ds'].dt.year.unique())
    holidays_df = pd.DataFrame(list(india_holidays.items()), columns=['ds', 'holiday'])
    return holidays_df

def predict_stock_prices(stock_data, years):
    # Prepare data for Prophet
    df = stock_data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

    # Add holidays
    holidays_df = add_holidays(df)
    
    # Create and fit the model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        holidays=holidays_df
    )
    
    # Add additional seasonality if needed
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    model.fit(df)
    
    # Create a DataFrame for future predictions
    future = model.make_future_dataframe(periods=years * 365)
    forecast = model.predict(future)
    
    return model, forecast

def plot_stock_data(stock_data, forecast, model, stock_symbol):
    # Plot actual and predicted stock prices
    fig = go.Figure()

    # Actual stock prices
    fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'], mode='lines', name='Actual Price', line=dict(color='red')))

    # Predicted stock prices
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted Price'))

    # Customizing the layout
    fig.update_layout(
        title=f'Stock Price Prediction for {stock_symbol}',
        xaxis_title='Date',
        yaxis_title='Price (INR)',
        hovermode='x',
        showlegend=True
    )
    
    # Adding interactivity for zooming and detailed information on hover
    fig.update_traces(marker=dict(size=5),
                      selector=dict(mode='markers+lines'))
    
    # Enable zooming and panning
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type="date"))

    return fig

def main():
    st.title('Stock Price Prediction App')
    
    stock_symbol = st.text_input("Enter the stock symbol (e.g., 'AAPL' for Apple):")
    years = st.number_input("Enter the number of years to predict:", min_value=1, max_value=10, value=1)
    
    if st.button("Predict"):
        if stock_symbol:
            st.write("Fetching data...")
            with st.spinner("Fetching stock data..."):
                stock_data = fetch_stock_data(stock_symbol)
            st.success("Data fetched successfully!")
            
            st.write("Predicting future prices...")
            with st.spinner("Predicting stock prices..."):
                model, forecast = predict_stock_prices(stock_data, years)
            st.success("Prediction completed!")
            
            fig = plot_stock_data(stock_data, forecast, model, stock_symbol)
            st.plotly_chart(fig)
        else:
            st.warning("Please enter a valid stock symbol.")

if __name__ == "__main__":
    main()
