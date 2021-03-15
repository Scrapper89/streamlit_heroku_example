import streamlit as st
import pandas as pd
import numpy as np
import datetime
import markowitz_example as mwe
import plotly.express as px
#st.set_page_config(layout='wide')

st.title('Markowitz Portfolio Optimization')

@st.cache
def load_data(choice):
    data = mwe.get_data(choice)
    return data

choice = st.radio("Dataset:", ('BTC','ETF'))
data_load_state = st.text('Loading data...')
data = load_data(choice)
data_load_state.text("Done! (using st.cache)")
opt_portfolios = None

#if st.button('Run Portfolio Optimization'):
st.subheader('Risk Adjusted PFolio')

@st.cache
def get_opt_portfolios(choice):
    df = load_data(choice)
    df = df[df.index > pd.to_datetime(datetime.date(2019, 1, 1))]
    opt_portfolios = mwe.generate_opt_portfolios(df, nq = 30, nrange = 500)
    opt_portfolios= opt_portfolios.sort_values('var').reset_index(drop=True)
    return df, opt_portfolios

st.subheader("Optimal VAR/Return Portfolios")
df, opt_portfolios = get_opt_portfolios(choice)
st.dataframe(opt_portfolios)

st.subheader("Portfolio Returns")
pfolios = mwe.generate_pfolios(df, opt_portfolios)
fig2 = px.line(pfolios)
st.plotly_chart( fig2, use_container_width=True )

st.subheader("Monthly Averages")
st.dataframe( mwe.get_monthly_returns(pfolios).T )

if choice=='BTC':
    st.header("Portfolio (BTC) Stress-Testing")
    st.subheader("2018 Crash")
    df1 = load_data(choice).copy()
    btc_crash_one = df1[
        (df1.index > pd.to_datetime(datetime.date(2018, 1, 1))) & (df1.index < pd.to_datetime(datetime.date(2019, 1, 1)))]
    btc_crash_one_pfolios = mwe.generate_pfolios(btc_crash_one, opt_portfolios)

    fig3 = px.line(btc_crash_one_pfolios)
    st.plotly_chart(fig3, use_container_width=True)
    st.dataframe(mwe.get_monthly_returns(btc_crash_one_pfolios).T)

    st.subheader("2019 Mini-Crash")
    btc_crash_two = df[
        (df.index > pd.to_datetime(datetime.date(2019, 7, 1))) & (df.index < pd.to_datetime(datetime.date(2020, 3, 1)))]
    btc_crash_two_pfolios = mwe.generate_pfolios(btc_crash_two, opt_portfolios)
    fig4 = px.line(btc_crash_two_pfolios)
    st.plotly_chart(fig4, use_container_width=True)
    st.dataframe(mwe.get_monthly_returns(btc_crash_two_pfolios).T)
