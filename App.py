import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import random

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Pahel AI Trading Dashboard",
    layout="wide",
    page_icon="📈"
)

# ==================================================
# UI STYLE
# ==================================================
st.markdown("""
<style>
.stApp { background-color:#0d1117; color:#c9d1d9; }
.main-title {
    font-size:38px;
    font-weight:800;
    color:#58a6ff;
    text-align:center;
}
.box {
    background:#161b22;
    padding:18px;
    border-radius:12px;
    border:1px solid #30363d;
}
.disclaimer {
    background:#1f2937;
    padding:15px;
    border-radius:12px;
    border-left:5px solid orange;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.title("⚙️ Trading Controls")

    coin = st.text_input("Crypto Symbol", "BTC").upper()
    ticker = f"{coin}-USD"

    history = st.selectbox(
        "Historical Data",
        ["1 Month", "6 Months", "1 Year", "5 Years"],
        index=2
    )

    future_type = st.selectbox("Prediction Type", ["Days", "Months", "Years"])
    future_value = st.slider("Future Duration", 1, 365, 7)

    st.sidebar.subheader("💱 Select Currency")
    currency = st.selectbox(
        "Currency",
        ["USD", "INR", "EUR", "GBP", "JPY"],
        index=0
    )

    st.info("📡 Data Source: Yahoo Finance")

# ==================================================
# PERIOD MAP
# ==================================================
period_map = {
    "1 Month": "1mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "5 Years": "5y"
}

# ==================================================
# LOAD DATA
# ==================================================
@st.cache_data(ttl=60)
def load_data(symbol, period):
    df = yf.download(symbol, period=period, progress=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

df = load_data(ticker, period_map[history])

# ==================================================
# MAIN APP
# ==================================================
if df is not None:

    st.markdown('<div class="main-title">₿ Pahel AI Trading Dashboard</div>', unsafe_allow_html=True)
    st.markdown("---")

    close = df["Close"].values
    live_price = float(close[-1])
    avg_price = float(np.mean(close))

    # ==================================================
    # USER PRICE INPUT
    # ==================================================
    st.subheader("💰 Enter Your Bitcoin Price")

    user_price = st.number_input(
        "Bitcoin Price (USD)",
        value=round(live_price, 2),
        step=50.0
    )

    # ==================================================
    # CURRENCY CONVERSION
    # ==================================================
    currency_rates = {
        "USD": 1,
        "INR": 83,
        "EUR": 0.92,
        "GBP": 0.81,
        "JPY": 145
    }

    currency_symbol = {
        "USD": "$",
        "INR": "₹",
        "EUR": "€",
        "GBP": "£",
        "JPY": "¥"
    }

    conversion_rate = currency_rates[currency]
    live_price_currency = live_price * conversion_rate
    user_price_currency = user_price * conversion_rate

    # ==================================================
    # ML MODEL
    # ==================================================
    model_df = pd.DataFrame({"price": close})
    model_df["index"] = np.arange(len(model_df))

    X = model_df[["index"]].values
    y = model_df["price"].values

    model = RandomForestRegressor(
        n_estimators=250,
        random_state=42
    )
    model.fit(X, y)

    # ==================================================
    # FUTURE DAYS
    # ==================================================
    if future_type == "Days":
        future_days = future_value
    elif future_type == "Months":
        future_days = future_value * 30
    else:
        future_days = future_value * 365

    future_index = np.array([[len(model_df) + future_days]])
    future_price = float(model.predict(future_index)[0])
    future_price_currency = future_price * conversion_rate
    avg_price_currency = avg_price * conversion_rate

    # ==================================================
    # PRICE CHANGE %
    # ==================================================
    change_pct = ((future_price - user_price) / user_price) * 100

    # ==================================================
    # TREND + SIGNAL
    # ==================================================
    if change_pct > 5:
        trend = "📈 Bullish"
        signal = "🟢 BUY"
    elif change_pct < -5:
        trend = "📉 Bearish"
        signal = "🔴 SELL"
    else:
        trend = "🟡 Sideways"
        signal = "🟡 HOLD"

    # ==================================================
    # MARKET ZONE
    # ==================================================
    if user_price > avg_price * 1.1:
        zone = "🔴 HIGH"
    elif user_price < avg_price * 0.9:
        zone = "🟢 LOW"
    else:
        zone = "🟡 MEDIUM"

    # ==================================================
    # AI BRAIN MODEL
    # ==================================================
    confidence = min(95, max(55, abs(change_pct) * 8))

    if confidence > 80 and trend == "📈 Bullish":
        brain = "🧠 AI Brain predicts strong upward momentum."
    elif confidence > 80 and trend == "📉 Bearish":
        brain = "🧠 AI Brain detects strong downside risk."
    else:
        brain = "🧠 AI Brain expects moderate volatility."

    # ==================================================
    # DASHBOARD METRICS
    # ==================================================
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Your Price", f"{currency_symbol[currency]}{user_price_currency:,.2f}")
    c2.metric("🔮 Future Price", f"{currency_symbol[currency]}{future_price_currency:,.2f}")
    c3.metric("📊 Market Zone", zone)
    c4.metric("📈 Trend", trend)

    # ==================================================
    # GRAPH
    # ==================================================
    st.subheader("📉 Bitcoin Market Chart")

    fig = go.Figure()

    fig.add_candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Market"
    )

    fig.add_trace(go.Scatter(
        x=[df.index[-1], df.index[-1] + pd.Timedelta(days=future_days)],
        y=[user_price, future_price],
        mode="lines+markers+text",
        text=["Your Price", f"{currency_symbol[currency]}{future_price_currency:,.2f}"],
        line=dict(color="lime", dash="dash"),
        name="Prediction"
    ))

    fig.update_layout(
        template="plotly_dark",
        height=550,
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==================================================
    # AI BRAIN CONFIDENCE GRAPH
    # ==================================================
    st.subheader("🧠 AI Brain Confidence Over Time")

    history_len = len(df)
    confidence_history = []

    for i in range(history_len):
        past_price = df["Close"].iloc[i]
        pct_change = ((close[-1] - past_price) / past_price) * 100
        conf = min(95, max(55, abs(pct_change) * 8))
        confidence_history.append(conf)

    conf_fig = go.Figure()
    conf_fig.add_trace(go.Scatter(
        x=df.index,
        y=confidence_history,
        mode='lines+markers',
        name='AI Confidence',
        line=dict(color='orange', width=3),
        marker=dict(size=4)
    ))
    conf_fig.update_layout(
        template="plotly_dark",
        height=400,
        yaxis_title="Confidence (%)",
        xaxis_title="Date",
        yaxis=dict(range=[50, 100])
    )

    st.plotly_chart(conf_fig, use_container_width=True)

    # ==================================================
    # PAHEL AI INTERACTIVE ASSISTANT
    # ==================================================
    st.subheader("💬 Pahel AI - Ask Anything!")
    user_question = st.text_input("Ask Pahel a question about Bitcoin or crypto...")

    def pahel_answer(question, user_price_currency, future_price_currency, trend, zone, confidence, change_pct, currency_symbol):
        q = question.lower()
        answer = "🤖 Pahel says: Sorry, I couldn't understand the question. Try asking about price, trend, buy/sell, or signals."

        if "price" in q or "future" in q:
            answer = f"💰 Current price: {currency_symbol}{user_price_currency:,.2f}, Predicted price: {currency_symbol}{future_price_currency:,.2f}."
        elif "trend" in q or "signal" in q:
            answer = f"📈 Trend: {trend}, Market Zone: {zone}, AI Confidence: {confidence:.0f}%."
        elif "buy" in q:
            if trend == "📈 Bullish":
                answer = "🟢 Pahel Advice: Market bullish, consider buying cautiously."
            elif trend == "📉 Bearish":
                answer = "🔴 Pahel Advice: Market bearish, avoid buying."
            else:
                answer = "🟡 Pahel Advice: Market sideways, monitor before buying."
        elif "sell" in q:
            if trend == "📉 Bearish":
                answer = "🔴 Pahel Advice: Market bearish, consider selling."
            elif trend == "📈 Bullish":
                answer = "🟢 Pahel Advice: Market bullish, consider holding instead of selling."
            else:
                answer = "🟡 Pahel Advice: Market sideways, selling not recommended."
        elif "hold" in q:
            answer = "🟡 Pahel Advice: Hold your position based on market trend and AI confidence."
        elif "risk" in q or "volatility" in q:
            if abs(change_pct) > 5:
                answer = "⚠️ Pahel Alert: Market volatility is high, trade carefully."
            else:
                answer = "✅ Pahel: Market volatility is moderate, generally safe."
        return answer

    if user_question:
        answer = pahel_answer(user_question, user_price_currency, future_price_currency, trend, zone, confidence, change_pct, currency_symbol)
        st.markdown(f"<div class='box'>{answer}</div>", unsafe_allow_html=True)

    # ==================================================
    # UNIQUE FEATURE 1: PAHEL SUGGESTION CARDS
    # ==================================================
    st.subheader("💡 Pahel Suggestions")
    bullish = min(90, max(10, 50 + change_pct * 2))  # needed for momentum
    suggestions = [
        f"Trend is {trend}, Market Zone: {zone}. AI Confidence: {confidence:.0f}%",
        f"Predicted change: {change_pct:.2f}% over next {future_value} {future_type.lower()}",
        "Monitor Bitcoin closely during high volatility periods.",
        "Use small position sizes when market is sideways.",
    ]

    cols = st.columns(len(suggestions))
    for i, s in enumerate(suggestions):
        cols[i].markdown(f"<div class='box'>{s}</div>", unsafe_allow_html=True)

    # ==================================================
    # UNIQUE FEATURE 2: CRYPTO VOLATILITY HEATMAP
    # ==================================================
    st.subheader("🌡 Volatility Heatmap (Last 30 Days)")
    recent_vol = df['Close'].pct_change().rolling(5).std() * 100
    heat_colors = ['#00FF00' if v<2 else '#FFFF00' if v<5 else '#FF4500' for v in recent_vol]

    heatmap_fig = go.Figure(data=go.Heatmap(
        z=[recent_vol.values],
        x=df.index[-len(recent_vol):],
        y=['Volatility'],
        colorscale=heat_colors,
        showscale=True
    ))
    heatmap_fig.update_layout(template='plotly_dark', height=200)
    st.plotly_chart(heatmap_fig, use_container_width=True)

    # ==================================================
    # UNIQUE FEATURE 3: MARKET MOMENTUM GAUGE
    # ==================================================
    st.subheader("📊 Market Momentum Gauge")
    momentum_score = (confidence + bullish) / 2
    if momentum_score > 75:
        momentum = "🟢 Strong Bullish Momentum"
    elif momentum_score > 50:
        momentum = "🟡 Neutral Momentum"
    else:
        momentum = "🔴 Bearish Momentum"

    st.metric("Momentum", momentum, delta=f"{momentum_score:.0f}%")

    # ==================================================
    # UNIQUE FEATURE 4: PAHEL AI CRYSTAL BALL 🔮
    # ==================================================
    st.subheader("🔮 Pahel AI Crystal Ball - Future Scenarios")
    if st.button("🔮 See Pahel AI Crystal Ball Prediction"):

        scenario_count = 5
        scenario_prices = []
        scenario_trends = []

        for _ in range(scenario_count):
            variation = random.uniform(-0.08, 0.08)  # simulate ±8% variation
            scen_price = future_price * (1 + variation)
            scenario_prices.append(scen_price * conversion_rate)
            if variation > 0.04:
                scenario_trends.append("📈 Bullish path looks sunny!")
            elif variation < -0.04:
                scenario_trends.append("📉 Bearish path, caution!")
            else:
                scenario_trends.append("🟡 Neutral path, monitor carefully.")

        for i in range(scenario_count):
            st.markdown(f"<div class='box'>Scenario {i+1}: {currency_symbol[currency]}{scenario_prices[i]:,.2f} - {scenario_trends[i]}</div>", unsafe_allow_html=True)

        # Optional: plot the crystal ball paths
        crystal_fig = go.Figure()
        crystal_fig.add_trace(go.Scatter(
            x=[df.index[-1]]*scenario_count,
            y=scenario_prices,
            mode='markers+lines',
            marker=dict(size=10, color='magenta'),
            line=dict(dash='dot'),
            name='Crystal Ball Paths'
        ))
        crystal_fig.update_layout(template='plotly_dark', height=400, yaxis_title=f"Price ({currency_symbol[currency]})")
        st.plotly_chart(crystal_fig, use_container_width=True)

    # ==================================================
    # SOCIAL SENTIMENT
    # ==================================================
    st.subheader("🌍 Community Sentiment")
    bearish = 100 - bullish
    st.progress(int(bullish))
    st.write(f"🟢 Bullish: {int(bullish)}% | 🔴 Bearish: {int(bearish)}%")

    # ==================================================
    # MARKET HEAT
    # ==================================================
    volatility = np.std(close[-20:]) / np.mean(close[-20:]) * 100
    if volatility > 8:
        heat = "🔥 Overheated Market"
    elif volatility > 4:
        heat = "🌤 Active Market"
    else:
        heat = "🧊 Cold Market"
    st.metric("🌡 Market Heat Index", heat)

    # ==================================================
    # DISCLAIMER
    # ==================================================
    st.markdown("""
    <div class="disclaimer">
    ⚠️ <b>Disclaimer</b><br><br>
    This project is for educational purposes only.<br>
    Cryptocurrency markets are highly volatile.<br>
    Predictions are trend-based using historical Yahoo Finance data.<br>
    This application does NOT provide financial advice.
    </div>
    """, unsafe_allow_html=True)

    st.success("✅ All features loaded successfully 🚀")

else:
    st.error("❌ Internet issue or invalid crypto symbol.")
