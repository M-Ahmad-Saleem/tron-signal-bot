"""
TRON (TRX) Spot Signal Bot
===========================
• Binance public API se real-time data
• GitHub Actions par cron schedule se chalta hai (system off ho toh bhi)
• Gmail se email alert bhejta hai
• State file se dobara same signal nahi bhejta (cooldown)

Environment Variables (GitHub Secrets mein set karo):
  GMAIL_SENDER    - aapki Gmail address (bot wali)
  GMAIL_PASSWORD  - Gmail App Password (16-char)
  GMAIL_RECEIVER  - jis email pe alert chahiye
"""

import os
import json
import smtplib
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("TRONBot")


# ── Config ────────────────────────────────────────────────────────────────────
class Config:
    # Gmail (GitHub Secrets se aata hai)
    GMAIL_SENDER    = os.environ.get("GMAIL_SENDER", "")
    GMAIL_PASSWORD  = os.environ.get("GMAIL_PASSWORD", "")
    GMAIL_RECEIVER  = os.environ.get("GMAIL_RECEIVER", "")

    # Trading
    SYMBOL   = "TRXUSDT"
    INTERVAL = "15m"
    CANDLES  = 100

    # Levels (Binance chart dekh ke update karo)
    RESISTANCE = 0.2950
    SUPPORT    = 0.2580

    # Indicators
    RSI_OVERSOLD   = 35.0
    RSI_OVERBOUGHT = 70.0
    RSI_BULL_ZONE  = 50.0
    VOLUME_SPIKE   = 1.5
    BREAKOUT_CANDLES = 2

    # Risk Management
    STOP_LOSS_PCT     = 5.0
    TAKE_PROFIT_1_PCT = 5.5
    TAKE_PROFIT_2_PCT = 13.0
    POSITION_SIZE_PCT = 5.0

    # Cooldown — state.json file se track hota hai
    COOLDOWN_HOURS = 4   # same signal kitne ghante baad dobara bheje

    # Minimum confidence to send alert
    MIN_CONFIDENCE = 55  # 0-100


cfg = Config()
STATE_FILE = Path("state.json")


# ── State Management (cooldown across GitHub Actions runs) ────────────────────
def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"last_signal": "", "last_signal_time": 0, "runs": 0}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def should_send(state: dict, signal: str, confidence: int) -> bool:
    if signal == "WAIT":
        return False
    if confidence < cfg.MIN_CONFIDENCE:
        log.info(f"Confidence {confidence}% < {cfg.MIN_CONFIDENCE}% threshold — skip")
        return False
    elapsed_hours = (datetime.now().timestamp() - state.get("last_signal_time", 0)) / 3600
    if signal == state.get("last_signal") and elapsed_hours < cfg.COOLDOWN_HOURS:
        remaining = cfg.COOLDOWN_HOURS - elapsed_hours
        log.info(f"Cooldown active — {remaining:.1f}h remaining for {signal}")
        return False
    return True


# ── Data Fetching — Multi-source fallback ─────────────────────────────────────
# Priority: Binance Global → Binance US → Bybit → KuCoin
# Agar ek block ho toh automatically next try karta hai

INTERVAL_MAP_BYBIT = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15,
    "30m": 30, "1h": 60, "2h": 120, "4h": 240, "1d": "D"
}
INTERVAL_MAP_KUCOIN = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min",
    "30m": "30min", "1h": "1hour", "2h": "2hour", "4h": "4hour", "1d": "1day"
}


def _klines_binance(symbol, interval, limit) -> pd.DataFrame:
    """Binance Global"""
    url = "https://api.binance.com/api/v3/klines"
    r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=12)
    r.raise_for_status()
    df = pd.DataFrame(r.json(), columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_vol","trades","taker_base","taker_quote","ignore"
    ])
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df


def _klines_binance_us(symbol, interval, limit) -> pd.DataFrame:
    """Binance US (different domain, less restricted)"""
    url = "https://api.binance.us/api/v3/klines"
    r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=12)
    r.raise_for_status()
    df = pd.DataFrame(r.json(), columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_vol","trades","taker_base","taker_quote","ignore"
    ])
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df


def _klines_bybit(symbol, interval, limit) -> pd.DataFrame:
    """Bybit — globally accessible"""
    iv = INTERVAL_MAP_BYBIT.get(interval, 15)
    url = "https://api.bybit.com/v5/market/kline"
    r = requests.get(url, params={
        "category": "spot", "symbol": symbol,
        "interval": str(iv), "limit": limit
    }, timeout=12)
    r.raise_for_status()
    data = r.json()
    if data.get("retCode") != 0:
        raise ValueError(f"Bybit error: {data.get('retMsg')}")
    rows = data["result"]["list"]
    df = pd.DataFrame(rows, columns=["open_time","open","high","low","close","volume","turnover"])
    df = df.iloc[::-1].reset_index(drop=True)  # oldest first
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"].astype(float), unit="ms")
    return df


def _klines_kucoin(symbol, interval, limit) -> pd.DataFrame:
    """KuCoin — globally accessible"""
    # KuCoin symbol format: TRX-USDT
    ks = symbol.replace("USDT", "-USDT").replace("BTC", "-BTC")
    iv = INTERVAL_MAP_KUCOIN.get(interval, "15min")
    url = "https://api.kucoin.com/api/v1/market/candles"
    r = requests.get(url, params={"symbol": ks, "type": iv}, timeout=12)
    r.raise_for_status()
    data = r.json()
    if data.get("code") != "200000":
        raise ValueError(f"KuCoin error: {data.get('msg')}")
    rows = data["data"][-limit:]
    rows = list(reversed(rows))  # oldest first
    df = pd.DataFrame(rows, columns=["open_time","open","close","high","low","volume","turnover"])
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"].astype(float), unit="s")
    return df


def fetch_klines(symbol, interval, limit=100) -> pd.DataFrame:
    """Try multiple exchanges until one works."""
    sources = [
        ("Binance Global", _klines_binance),
        ("Binance US",     _klines_binance_us),
        ("Bybit",          _klines_bybit),
        ("KuCoin",         _klines_kucoin),
    ]
    last_err = None
    for name, fn in sources:
        try:
            df = fn(symbol, interval, limit)
            log.info(f"Data source: {name} ✓")
            return df
        except Exception as e:
            log.warning(f"{name} failed: {e}")
            last_err = e
    raise RuntimeError(f"Sab data sources fail ho gaye. Last error: {last_err}")


def fetch_ticker(symbol) -> dict:
    """24h ticker — try Binance, fallback to Bybit, then KuCoin."""
    # Binance Global
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/24hr",
                         params={"symbol": symbol}, timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning(f"Binance ticker failed: {e}")

    # Binance US
    try:
        r = requests.get("https://api.binance.us/api/v3/ticker/24hr",
                         params={"symbol": symbol}, timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning(f"Binance US ticker failed: {e}")

    # Bybit fallback
    try:
        r = requests.get("https://api.bybit.com/v5/market/tickers",
                         params={"category": "spot", "symbol": symbol}, timeout=12)
        r.raise_for_status()
        t = r.json()["result"]["list"][0]
        return {
            "lastPrice":          t.get("lastPrice", "0"),
            "priceChangePercent": t.get("price24hPcnt", "0"),
            "quoteVolume":        t.get("turnover24h", "0"),
        }
    except Exception as e:
        log.warning(f"Bybit ticker failed: {e}")

    # KuCoin fallback
    try:
        ks = symbol.replace("USDT", "-USDT")
        r = requests.get(f"https://api.kucoin.com/api/v1/market/stats",
                         params={"symbol": ks}, timeout=12)
        r.raise_for_status()
        t = r.json()["data"]
        change_pct = float(t.get("changeRate", 0)) * 100
        return {
            "lastPrice":          str(t.get("last", "0")),
            "priceChangePercent": str(round(change_pct, 2)),
            "quoteVolume":        str(t.get("volValue", "0")),
        }
    except Exception as e:
        log.warning(f"KuCoin ticker failed: {e}")

    # Last resort — return dummy ticker so analysis can still run
    log.warning("Ticker fetch failed from all sources — using dummy values")
    return {"lastPrice": "0", "priceChangePercent": "0", "quoteVolume": "0"}


# ── Technical Indicators ──────────────────────────────────────────────────────
def calc_rsi(s: pd.Series, p=14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).ewm(com=p-1, min_periods=p).mean()
    l = (-d).clip(lower=0).ewm(com=p-1, min_periods=p).mean()
    return 100 - 100 / (1 + g / l)


def calc_macd(s: pd.Series, fast=12, slow=26, sig=9):
    ml = s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()
    sl = ml.ewm(span=sig, adjust=False).mean()
    return ml, sl, ml - sl


def calc_bb(s: pd.Series, p=20, k=2):
    sma = s.rolling(p).mean()
    std = s.rolling(p).std()
    return sma + k*std, sma, sma - k*std


def calc_ema(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(span=p, adjust=False).mean()


def calc_stoch(df: pd.DataFrame, k=14, d=3):
    lo = df["low"].rolling(k).min()
    hi = df["high"].rolling(k).max()
    kp = 100 * (df["close"] - lo) / (hi - lo)
    return kp, kp.rolling(d).mean()


# ── Analysis ──────────────────────────────────────────────────────────────────
class Signal:
    def __init__(self):
        self.signal = "WAIT"
        self.confidence = 0
        self.reasons = []
        self.warnings = []
        self.price = 0.0
        self.rsi = 0.0
        self.macd = 0.0
        self.vol_ratio = 0.0
        self.ema_trend = ""
        self.bb_pos = ""
        self.stoch_k = 0.0
        self.stoch_d = 0.0
        self.breakout = False
        self.change_24h = 0.0
        self.volume_24h = 0.0
        self.stop_loss = 0.0
        self.tp1 = 0.0
        self.tp2 = 0.0


def analyze(df: pd.DataFrame, ticker: dict) -> Signal:
    s = Signal()
    c = df["close"]
    score = 0
    max_s = 0

    s.price     = c.iloc[-1]
    s.change_24h = float(ticker.get("priceChangePercent", 0))
    s.volume_24h = float(ticker.get("quoteVolume", 0))
    s.stop_loss  = round(s.price * (1 - cfg.STOP_LOSS_PCT / 100), 6)
    s.tp1        = round(s.price * (1 + cfg.TAKE_PROFIT_1_PCT / 100), 6)
    s.tp2        = round(s.price * (1 + cfg.TAKE_PROFIT_2_PCT / 100), 6)

    # RSI
    rsi_s = calc_rsi(c)
    s.rsi = round(rsi_s.iloc[-1], 2)
    max_s += 25
    if s.rsi < cfg.RSI_OVERSOLD:
        score += 25; s.reasons.append(f"RSI {s.rsi} — oversold (strong buy zone)")
    elif s.rsi >= cfg.RSI_BULL_ZONE and s.rsi < cfg.RSI_OVERBOUGHT:
        score += 15; s.reasons.append(f"RSI {s.rsi} — bullish zone")
    elif s.rsi >= cfg.RSI_OVERBOUGHT:
        score -= 15; s.warnings.append(f"RSI {s.rsi} — overbought, risky entry")

    # MACD
    ml, sl, hist = calc_macd(c)
    s.macd = round(ml.iloc[-1], 6)
    max_s += 25
    if hist.iloc[-1] > 0 and hist.iloc[-2] <= 0:
        score += 25; s.reasons.append("MACD bullish crossover — strong entry signal")
    elif hist.iloc[-1] > 0:
        score += 15; s.reasons.append("MACD positive — uptrend mein hai")
    elif hist.iloc[-1] < 0 and hist.iloc[-2] >= 0:
        score -= 25; s.warnings.append("MACD bearish crossover — downtrend shuru")
    else:
        score -= 10; s.warnings.append("MACD negative — downtrend")

    # Bollinger Bands
    upper, mid, lower = calc_bb(c)
    max_s += 15
    if s.price <= lower.iloc[-1]:
        score += 15; s.bb_pos = "Lower band"; s.reasons.append("Price lower BB — reversal possible")
    elif s.price >= upper.iloc[-1]:
        score -= 15; s.bb_pos = "Upper band"; s.warnings.append("Price upper BB — overbought")
    else:
        bw = upper.iloc[-1] - lower.iloc[-1]
        pos = (s.price - lower.iloc[-1]) / bw if bw > 0 else 0.5
        if pos < 0.4:
            score += 8; s.bb_pos = "Lower half (bullish)"; s.reasons.append("BB lower half — mild bullish")
        else:
            s.bb_pos = "Mid zone (neutral)"

    # EMA 20 / 50
    e20 = calc_ema(c, 20); e50 = calc_ema(c, 50)
    max_s += 20
    if e20.iloc[-1] > e50.iloc[-1] and e20.iloc[-2] <= e50.iloc[-2]:
        score += 20; s.ema_trend = "Golden cross ↑"; s.reasons.append("EMA golden cross — strong uptrend")
    elif e20.iloc[-1] > e50.iloc[-1]:
        score += 12; s.ema_trend = "Uptrend ↑"; s.reasons.append("EMA20 > EMA50 — uptrend")
    elif e20.iloc[-1] < e50.iloc[-1] and e20.iloc[-2] >= e50.iloc[-2]:
        score -= 20; s.ema_trend = "Death cross ↓"; s.warnings.append("EMA death cross — downtrend")
    else:
        score -= 8; s.ema_trend = "Downtrend ↓"; s.warnings.append("EMA bearish")

    # Volume
    avg_v = df["volume"].iloc[-20:].mean()
    cur_v = df["volume"].iloc[-1]
    s.vol_ratio = round(cur_v / avg_v, 2) if avg_v > 0 else 1.0
    max_s += 15
    if s.vol_ratio >= cfg.VOLUME_SPIKE:
        score += 15; s.reasons.append(f"Volume spike {s.vol_ratio}x — move confirm")
    elif s.vol_ratio >= 1.2:
        score += 8; s.reasons.append(f"Volume {s.vol_ratio}x — mild confirm")
    else:
        s.warnings.append(f"Volume low {s.vol_ratio}x — signal weak ho sakta hai")

    # Stochastic
    sk, sd = calc_stoch(df)
    s.stoch_k = round(sk.iloc[-1], 1)
    s.stoch_d = round(sd.iloc[-1], 1)
    if s.stoch_k < 20 and s.stoch_k > s.stoch_d:
        s.reasons.append(f"Stoch oversold + bullish ({s.stoch_k})")
    elif s.stoch_k > 80:
        s.warnings.append(f"Stoch overbought ({s.stoch_k})")

    # Breakout
    last_n = df["close"].iloc[-cfg.BREAKOUT_CANDLES:]
    if all(last_n > cfg.RESISTANCE):
        s.breakout = True
        score += 20
        s.reasons.append(f"BREAKOUT! ${cfg.RESISTANCE} resistance cross ho gaya!")

    pct = (score / max_s * 100) if max_s > 0 else 0
    s.confidence = max(0, min(100, int(pct)))

    if s.breakout or score >= max_s * 0.65:
        s.signal = "BUY"
    elif score <= max_s * 0.25:
        s.signal = "SELL"
    else:
        s.signal = "WAIT"

    return s


# ── Gmail Sender ──────────────────────────────────────────────────────────────
def send_gmail(subject: str, html_body: str) -> bool:
    if not all([cfg.GMAIL_SENDER, cfg.GMAIL_PASSWORD, cfg.GMAIL_RECEIVER]):
        log.error("Gmail credentials missing! GMAIL_SENDER, GMAIL_PASSWORD, GMAIL_RECEIVER set karo.")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"TRON Signal Bot <{cfg.GMAIL_SENDER}>"
    msg["To"]      = cfg.GMAIL_RECEIVER
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(cfg.GMAIL_SENDER, cfg.GMAIL_PASSWORD)
            server.sendmail(cfg.GMAIL_SENDER, cfg.GMAIL_RECEIVER, msg.as_string())
        log.info(f"Email bheja: {cfg.GMAIL_RECEIVER}")
        return True
    except Exception as e:
        log.error(f"Gmail error: {e}")
        return False


def build_email(sig: Signal) -> tuple[str, str]:
    emoji   = {"BUY": "📈", "SELL": "📉", "WAIT": "⏳"}[sig.signal]
    color   = {"BUY": "#16a34a", "SELL": "#dc2626", "WAIT": "#d97706"}[sig.signal]
    label   = {"BUY": "BUY — Kharido!", "SELL": "SELL — Bacho!", "WAIT": "WAIT — Ruko"}[sig.signal]
    now     = datetime.now().strftime("%d %b %Y  %H:%M UTC")

    subject = f"{emoji} TRX {sig.signal} Signal — ${sig.price:.6f} | Confidence {sig.confidence}%"
    if sig.breakout:
        subject = f"🚨 BREAKOUT! {subject}"

    reasons_html  = "".join(f"<li style='margin:4px 0'>✅ {r}</li>" for r in sig.reasons)  if sig.reasons  else "<li>—</li>"
    warnings_html = "".join(f"<li style='margin:4px 0'>⚠️ {w}</li>" for w in sig.warnings) if sig.warnings else ""

    risk_block = ""
    if sig.signal == "BUY":
        risk_block = f"""
        <table style="width:100%;border-collapse:collapse;margin-top:16px">
          <tr style="background:#f0fdf4"><td style="padding:8px 12px;font-weight:600;color:#15803d">🟢 Take Profit 2</td><td style="padding:8px 12px;color:#15803d;text-align:right">${sig.tp2:.6f} &nbsp;(+{cfg.TAKE_PROFIT_2_PCT}%)</td></tr>
          <tr style="background:#f0fdf4"><td style="padding:8px 12px;font-weight:600;color:#15803d">🟢 Take Profit 1</td><td style="padding:8px 12px;color:#15803d;text-align:right">${sig.tp1:.6f} &nbsp;(+{cfg.TAKE_PROFIT_1_PCT}%)</td></tr>
          <tr style="background:#fff7ed"><td style="padding:8px 12px;font-weight:600">💰 Entry (current)</td><td style="padding:8px 12px;text-align:right">${sig.price:.6f}</td></tr>
          <tr style="background:#fef2f2"><td style="padding:8px 12px;font-weight:600;color:#dc2626">🔴 Stop Loss</td><td style="padding:8px 12px;color:#dc2626;text-align:right">${sig.stop_loss:.6f} &nbsp;(-{cfg.STOP_LOSS_PCT}%)</td></tr>
        </table>
        <p style="margin:12px 0 0;font-size:13px;color:#6b7280">📦 Position size: apni capital ka <strong>{cfg.POSITION_SIZE_PCT}%</strong> lagao</p>
        """

    breakout_banner = ""
    if sig.breakout:
        breakout_banner = f"""
        <div style="background:#fef9c3;border:2px solid #ca8a04;border-radius:8px;padding:12px 16px;margin-bottom:16px">
          🚨 <strong>BREAKOUT DETECTED!</strong> Price ${cfg.RESISTANCE} resistance ke upar {cfg.BREAKOUT_CANDLES} candles se close hua!
        </div>
        """

    warnings_section = ""
    if warnings_html:
        warnings_section = f"""
        <div style="margin-top:16px">
          <p style="font-weight:600;margin:0 0 6px">⚠️ Warnings:</p>
          <ul style="margin:0;padding-left:20px;color:#92400e">{warnings_html}</ul>
        </div>
        """

    html = f"""
<!DOCTYPE html><html><body style="margin:0;padding:0;background:#f3f4f6;font-family:Arial,sans-serif">
<div style="max-width:560px;margin:32px auto;background:#fff;border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.1)">

  <!-- Header -->
  <div style="background:{color};padding:24px;text-align:center">
    <p style="margin:0;font-size:32px">{emoji}</p>
    <h1 style="margin:8px 0 4px;color:#fff;font-size:22px">TRON (TRX/USDT)</h1>
    <p style="margin:0;color:rgba(255,255,255,0.9);font-size:18px;font-weight:700">{label}</p>
    <p style="margin:8px 0 0;color:rgba(255,255,255,0.8);font-size:13px">{now}</p>
  </div>

  <!-- Body -->
  <div style="padding:24px">
    {breakout_banner}

    <!-- Price stats -->
    <div style="display:flex;gap:12px;margin-bottom:20px">
      <div style="flex:1;background:#f9fafb;border-radius:8px;padding:12px;text-align:center">
        <p style="margin:0;font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.05em">Price</p>
        <p style="margin:4px 0 0;font-size:20px;font-weight:700">${sig.price:.6f}</p>
      </div>
      <div style="flex:1;background:#f9fafb;border-radius:8px;padding:12px;text-align:center">
        <p style="margin:0;font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.05em">24h Change</p>
        <p style="margin:4px 0 0;font-size:20px;font-weight:700;color:{'#16a34a' if sig.change_24h >= 0 else '#dc2626'}">{sig.change_24h:+.2f}%</p>
      </div>
      <div style="flex:1;background:#f9fafb;border-radius:8px;padding:12px;text-align:center">
        <p style="margin:0;font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.05em">Confidence</p>
        <p style="margin:4px 0 0;font-size:20px;font-weight:700;color:{color}">{sig.confidence}%</p>
      </div>
    </div>

    <!-- Indicators -->
    <p style="font-weight:700;margin:0 0 10px;font-size:15px">📐 Technical Indicators</p>
    <table style="width:100%;border-collapse:collapse;font-size:14px">
      <tr style="border-bottom:1px solid #e5e7eb"><td style="padding:7px 0;color:#6b7280">RSI (14)</td><td style="padding:7px 0;text-align:right;font-weight:600">{sig.rsi}</td></tr>
      <tr style="border-bottom:1px solid #e5e7eb"><td style="padding:7px 0;color:#6b7280">MACD</td><td style="padding:7px 0;text-align:right;font-weight:600;color:{'#16a34a' if sig.macd > 0 else '#dc2626'}">{sig.macd:+.6f}</td></tr>
      <tr style="border-bottom:1px solid #e5e7eb"><td style="padding:7px 0;color:#6b7280">Volume</td><td style="padding:7px 0;text-align:right;font-weight:600">{sig.vol_ratio}x average</td></tr>
      <tr style="border-bottom:1px solid #e5e7eb"><td style="padding:7px 0;color:#6b7280">EMA Trend</td><td style="padding:7px 0;text-align:right;font-weight:600">{sig.ema_trend}</td></tr>
      <tr style="border-bottom:1px solid #e5e7eb"><td style="padding:7px 0;color:#6b7280">Bollinger Bands</td><td style="padding:7px 0;text-align:right;font-weight:600">{sig.bb_pos}</td></tr>
      <tr><td style="padding:7px 0;color:#6b7280">Stoch K / D</td><td style="padding:7px 0;text-align:right;font-weight:600">{sig.stoch_k} / {sig.stoch_d}</td></tr>
    </table>

    <!-- Reasons -->
    <div style="margin-top:20px;padding:14px;background:#f0fdf4;border-radius:8px;border-left:4px solid #16a34a">
      <p style="font-weight:600;margin:0 0 8px">✅ Signal Reasons:</p>
      <ul style="margin:0;padding-left:20px;color:#166534">{reasons_html}</ul>
    </div>

    {warnings_section}

    <!-- Risk Management -->
    {risk_block}

    <!-- Levels -->
    <div style="margin-top:20px;display:flex;gap:10px">
      <div style="flex:1;background:#f0fdf4;border-radius:8px;padding:10px;text-align:center">
        <p style="margin:0;font-size:11px;color:#6b7280">Support</p>
        <p style="margin:4px 0 0;font-weight:700;color:#16a34a">${cfg.SUPPORT}</p>
      </div>
      <div style="flex:1;background:#fef2f2;border-radius:8px;padding:10px;text-align:center">
        <p style="margin:0;font-size:11px;color:#6b7280">Resistance</p>
        <p style="margin:4px 0 0;font-weight:700;color:#dc2626">${cfg.RESISTANCE}</p>
      </div>
    </div>
  </div>

  <!-- Footer -->
  <div style="padding:16px 24px;background:#f9fafb;border-top:1px solid #e5e7eb;text-align:center">
    <p style="margin:0;font-size:12px;color:#9ca3af">TRON Signal Bot • Binance TRX/USDT • {cfg.INTERVAL} chart</p>
    <p style="margin:6px 0 0;font-size:11px;color:#d1d5db">⚠️ Educational purposes only. Apna financial risk khud manage karein.</p>
  </div>
</div>
</body></html>
"""
    return subject, html


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 55)
    log.info("  TRON Signal Bot — GitHub Actions Run")
    log.info(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    log.info("=" * 55)

    state = load_state()
    state["runs"] = state.get("runs", 0) + 1
    log.info(f"Total runs so far: {state['runs']}")

    # Fetch data
    log.info(f"Binance se data fetch ho raha hai: {cfg.SYMBOL} {cfg.INTERVAL}")
    df     = fetch_klines(cfg.SYMBOL, cfg.INTERVAL, cfg.CANDLES)
    ticker = fetch_ticker(cfg.SYMBOL)
    log.info(f"Data ready: {len(df)} candles, price=${float(ticker['lastPrice']):.6f}")

    # Analyze
    sig = analyze(df, ticker)
    log.info(
        f"Signal={sig.signal} | Confidence={sig.confidence}% | "
        f"RSI={sig.rsi} | MACD={sig.macd:+.6f} | Vol={sig.vol_ratio}x | "
        f"Breakout={'YES' if sig.breakout else 'No'}"
    )

    # Send email?
    if should_send(state, sig.signal, sig.confidence):
        subject, html = build_email(sig)
        ok = send_gmail(subject, html)
        if ok:
            state["last_signal"]      = sig.signal
            state["last_signal_time"] = datetime.now().timestamp()
            log.info(f"Email sent! Signal: {sig.signal}")
        else:
            log.error("Email send nahi hua!")
    else:
        log.info(f"Email nahi bheja — signal={sig.signal}, confidence={sig.confidence}%")

    save_state(state)
    log.info("Run complete.")


if __name__ == "__main__":
    main()
