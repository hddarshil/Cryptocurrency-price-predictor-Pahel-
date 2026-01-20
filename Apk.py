import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor

# --- PAGE CONFIG ---
st.set_page_config(page_title="Bitcoin Price Prediction", layout="wide", page_icon="📈")

# Custom CSS
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .main-title { font-size: 38px; font-weight: 800; color: #58a6ff; text-align: center; }
    .stMetric { background: rgba(88, 166, 255, 0.05); border: 1px solid #30363d; border-radius: 12px; padding: 15px; }
    .stButton>button { width: 100%; background-color: #238636; color: white; border-radius: 8px; font-weight: bold; height: 3em; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    user_input = st.text_input("Enter Crypto Symbol (e.g. BTC, ETH)", "BTC").upper().strip()
    
    # Auto-fix Ticker for Yahoo Finance
    ticker = f"{user_input}-USD" if "-" not in user_input else user_input
    
    time_frame = st.selectbox("Select History", ["1 Week", "1 Month", "1 Year", "5 Years"], index=1)
    predict_days = st.slider("Prediction Days", 1, 30, 7)
    
    st.divider()
    st.info("Yahoo Finance Live Data Integrated")
    st.info("DISCLAIMER it Was Predict Form Past Data ! Eduction & Learning Purposes ! This Systeam Not Provaide a investment or trading advice") 
    st.info("The Developer is Not Responsible for any Bit Coin & Treaders User are advised to conduct their own research before making any investment decisions")   

# Mapping
p_map = {"1 Week": "7d", "1 Month": "1mo", "1 Year": "1y", "5 Years": "5y"}
i_map = {"1 Week": "1h", "1 Month": "1d", "1 Year": "1d", "5 Years": "1wk"}

# --- DATA FETCH ---
@st.cache_data(ttl=60)
def fetch_data(symbol, p, i):
    try:
        data = yf.download(symbol, period=p, interval=i, progress=False)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except: return None

df = fetch_data(ticker, p_map[time_frame], i_map[time_frame])

# --- MAIN UI ---
if df is not None:
    st.markdown('<div class="main-title">Crypto Currency Bitcoin Price Prediction</div>', unsafe_allow_html=True)
    st.write(f"<center>Analyzing Real-time Market Data for <b>{ticker}</b></center>", unsafe_allow_html=True)
    st.markdown("---")

    # Metrics
    curr_price = float(df['Close'].iloc[-1])
    high_val = float(df['High'].max())
    low_val = float(df['Low'].min())
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Live Price", f"${curr_price:,.2f}")
    m2.metric("Period High", f"${high_val:,.2f}")
    m3.metric("Period Low", f"${low_val:,.2f}")

    # Layout: Chart on Left, Prediction on Right
    col_chart, col_pred = st.columns([2, 1])

    with col_chart:
        st.subheader("📊 Market Trend")
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
        fig.update_layout(template="plotly_dark", height=450, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_pred:
        st.subheader("🔮 AI Projection")
        st.write("Click the button below to run the PAHEL AI Brain.")
        
        # --- PREDICTION BUTTON ---
        if st.button("Generate AI Prediction"):
            with st.spinner("AI is analyzing patterns..."):
                # ML Logic
                model_df = df[['Close']].copy().reset_index()
                model_df['Idx'] = np.arange(len(model_df))
                
                X = model_df[['Idx']].values
                y = model_df['Close'].values
                
                rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
                
                future_idx = np.array([len(model_df) + i for i in range(predict_days)]).reshape(-1, 1)
                preds = rf.predict(future_idx)
                
                # Small Forecast Chart
                f_fig = go.Figure()
                f_fig.add_trace(go.Scatter(y=df['Close'].tail(30), name="Recent", line=dict(color='#58a6ff')))
                f_fig.add_trace(go.Scatter(x=list(range(30, 30+predict_days)), y=preds, name="AI Forecast", line=dict(color='#39d353', dash='dash')))
                f_fig.update_layout(template="plotly_dark", height=300, showlegend=False)
                st.plotly_chart(f_fig, use_container_width=True)
                
                st.success(f"Expected Price: **${preds[-1]:,.2f}**")
                st.caption(f"Based on {ticker} history patterns.")
        else:
            st.info("Waiting for your command to predict.")

    # Export
    csv = df.to_csv().encode('utf-8')
    st.sidebar.download_button("📥 Download Market Report", data=csv, file_name=f"{ticker}_data.csv")

else:
    st.error(f"Invalid Ticker '{ticker}'. Make sure your internet is on.")
    