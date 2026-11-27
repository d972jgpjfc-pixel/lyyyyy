from datetime import datetime
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="股票基本面、互動波動圖與 AI 買點分析",
    layout="wide",
)
st.title("📈 股票基本面 × 互動波動圖 × AI 買點分析")

# --- 側邊欄參數設定 ---
with st.sidebar:
  st.header("⚙️ 參數設定")
  tickers_input = st.text_input(
      "股票代號（以逗號分隔）",
      value="2330.TW, SPY, NVDA",
  )
  start_date = st.date_input("歷史區間開始日期", value=datetime(2024, 1, 1))
  use_today = st.checkbox("歷史區間結束日期使用今天", value=True)
  custom_end_date = st.date_input(
      "自訂歷史區間結束日期",
      value=datetime.today(),
      disabled=use_today,
  )
  window = st.number_input(
      "移動平均天數 (Window)",
      min_value=5,
      max_value=500,
      value=252,
  )
  intraday_interval = st.selectbox(
      "當日波動圖時間顆粒",
      options=["5m", "1m", "15m", "30m"],
      index=0,
      help="5m 適合觀察全日波動節奏；1m 細節最細但資料點較多。",
  )
  amplitude_threshold = st.number_input(
      "當日振幅提醒門檻 (%)",
      min_value=0.5,
      max_value=30.0,
      value=3.0,
      step=0.5,
      help="當（當日最高價 - 當日最低價）/ 當日最低價 達此百分比時，顯示盤中波動提醒。",
  )
  run_btn = st.button(
      "🚀 開始分析",
      type="primary",
      use_container_width=True,
  )

  st.divider()
  st.header("⏱️ 盤中自動刷新")
  auto_refresh = st.checkbox(
      "啟用自動刷新（定期更新當日波動圖）",
      value=False,
      help="開啟後會依設定間隔自動重新抓取各檔股票當日分時資料並更新畫面。",
  )
  refresh_interval = st.number_input(
      "刷新間隔（秒）",
      min_value=15,
      max_value=600,
      value=60,
      step=15,
      disabled=not auto_refresh,
  )


@st.cache_data(ttl=3600)
def download_history_with_retry(
    symbol: str, start: str, end: str, retries: int = 3
) -> pd.DataFrame:
  """安全下載歷史股價，含重試與 MultiIndex 欄位處理"""
  for attempt in range(1, retries + 1):
    try:
      data = yf.download(
          symbol,
          start=start,
          end=end,
          progress=False,
          auto_adjust=False,
          threads=False,
      )
      if data is not None and not data.empty:
        if isinstance(data.columns, pd.MultiIndex):
          data.columns = data.columns.get_level_values(0)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
          if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data = data.dropna(subset=["High", "Low", "Close"])
        if not data.empty:
          return data
    except Exception:
      pass
    if attempt < retries:
      time.sleep(2)
  return pd.DataFrame()


@st.cache_data(ttl=60)
def download_intraday_data(
    symbol: str, interval: str = "5m"
) -> tuple[pd.DataFrame, str]:
  """下載當日分時波動資料；若無當日資料則自動改抓近 5 日分時作備援"""
  attempts = [
      ("1d", interval, "當日盤中/最近交易日分時"),
      ("5d", "15m" if interval in ("1m", "5m") else interval, "近 5 日分時（備援）"),
  ]
  for period, iv, label in attempts:
    try:
      df = yf.download(
          symbol,
          period=period,
          interval=iv,
          progress=False,
          auto_adjust=False,
          threads=False,
      )
      if df is not None and not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
          df.columns = df.columns.get_level_values(0)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
          if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["High", "Low", "Close"])
        if not df.empty:
          if period == "1d":
            last_day = df.index[-1].date()
            day_df = df[df.index.date == last_day]
            if not day_df.empty:
              return day_df, f"{label}（{last_day}，間隔 {iv}）"
          return df, f"{label}（間隔 {iv}）"
    except Exception:
      continue
  return pd.DataFrame(), "無分時資料"


@st.cache_data(ttl=3600)
def get_fundamental_info(ticker_symbol: str) -> dict | None:
  """抓取單一股票基本面指標與公司名稱"""
  try:
    stock = yf.Ticker(ticker_symbol)
    info = stock.info or {}
    bs = stock.balance_sheet

    if not info and (bs is None or bs.empty):
      return {
          "company_name": ticker_symbol,
          "col_title": ticker_symbol,
          "metrics": ["無資料"] * 7,
          "raw": {},
      }

    company_name = (
        info.get("longName")
        or info.get("shortName")
        or ticker_symbol
    )
    short_name = info.get("shortName") or company_name

    currency = info.get("currency", "")
    sector = info.get("sector", "N/A")
    industry = info.get("industry", "N/A")
    company_nature = (
        f"{sector} - {industry}"
        if (sector and sector != "N/A") or (industry and industry != "N/A")
        else "無資料"
    )

    eps = info.get("trailingEps", None)
    eps_str = f"{eps:.2f} ({currency})" if isinstance(eps, (int, float)) else "N/A"

    pe_ratio = info.get("trailingPE", None)
    pe_str = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else "N/A"

    gross_margin = info.get("grossMargins", None)
    gm_str = f"{gross_margin * 100:.2f}%" if isinstance(gross_margin, (int, float)) else "N/A"

    profit_margin = info.get("profitMargins", None)
    pm_str = f"{profit_margin * 100:.2f}%" if isinstance(profit_margin, (int, float)) else "N/A"

    ca_str = "N/A"
    debt_ratio_val = None
    debt_ratio_str = "N/A"
    if bs is not None and not bs.empty:
      latest_bs = bs.iloc[:, 0]
      if "Current Assets" in latest_bs and pd.notna(latest_bs["Current Assets"]):
        ca_val = float(latest_bs["Current Assets"])
        ca_str = f"{ca_val / 1e8:,.2f} 億 ({currency})"
      if (
          "Total Liabilities Net Minority Interest" in latest_bs
          and "Total Assets" in latest_bs
      ):
        total_liab = latest_bs["Total Liabilities Net Minority Interest"]
        total_assets = latest_bs["Total Assets"]
        if (
            pd.notna(total_liab)
            and pd.notna(total_assets)
            and float(total_assets) != 0
        ):
          debt_ratio_val = (float(total_liab) / float(total_assets)) * 100
          debt_ratio_str = f"{debt_ratio_val:.2f}%"

    col_title = f"{ticker_symbol} ({short_name})" if short_name else ticker_symbol

    return {
        "company_name": company_name,
        "col_title": col_title,
        "metrics": [
            company_nature,
            eps_str,
            pe_str,
            gm_str,
            pm_str,
            ca_str,
            debt_ratio_str,
        ],
        "raw": {
            "pe": pe_ratio if isinstance(pe_ratio, (int, float)) else None,
            "gross_margin_pct": (gross_margin * 100) if isinstance(gross_margin, (int, float)) else None,
            "profit_margin_pct": (profit_margin * 100) if isinstance(profit_margin, (int, float)) else None,
            "debt_ratio_pct": debt_ratio_val,
        },
    }
  except Exception:
    return {
        "company_name": ticker_symbol,
        "col_title": ticker_symbol,
        "metrics": ["無資料"] * 7,
        "raw": {},
    }


def compute_analysis(df: pd.DataFrame, window: int) -> tuple[dict, pd.DataFrame]:
  """計算歷史極值、布林通道指標與單檔分析摘要"""
  work_df = df.copy()
  absolute_high = float(work_df["High"].max())
  absolute_low = float(work_df["Low"].min())
  high_date = work_df["High"].idxmax().strftime("%Y-%m-%d")
  low_date = work_df["Low"].idxmin().strftime("%Y-%m-%d")
  current_price = float(work_df["Close"].iloc[-1])

  has_enough_window = len(work_df) >= window
  if has_enough_window:
    work_df["SMA"] = work_df["Close"].rolling(window=window).mean()
    work_df["STD"] = work_df["Close"].rolling(window=window).std()
    work_df["Estimated_High"] = work_df["SMA"] + (2 * work_df["STD"])
    work_df["Estimated_Low"] = work_df["SMA"] - (2 * work_df["STD"])

    latest_sma_val = float(work_df["SMA"].iloc[-1])
    latest_est_high_val = float(work_df["Estimated_High"].iloc[-1])
    latest_est_low_val = float(work_df["Estimated_Low"].iloc[-1])

    latest_sma = f"{latest_sma_val:.2f}"
    latest_est_high = f"{latest_est_high_val:.2f}"
    latest_est_low = f"{latest_est_low_val:.2f}"
  else:
    latest_sma_val = None
    latest_est_high_val = None
    latest_est_low_val = None
    latest_sma = "資料不足"
    latest_est_high = "資料不足"
    latest_est_low = "資料不足"

  summary = {
      "最新收盤價": f"{current_price:.2f}",
      "區間最高價": f"{absolute_high:.2f}",
      "區間最高價日期": high_date,
      "區間最低價": f"{absolute_low:.2f}",
      "區間最低價日期": low_date,
      f"基準均價 (SMA {window}日)": latest_sma,
      "預估壓力高點 (+2SD)": latest_est_high,
      "預估支撐低點 (-2SD)": latest_est_low,
      "has_enough_window": has_enough_window,
      "current_price_val": current_price,
      "latest_sma_val": latest_sma_val,
      "latest_est_high_val": latest_est_high_val,
      "latest_est_low_val": latest_est_low_val,
  }
  return summary, work_df


def _resolve_selected_index(event, labels: list[str], total_len: int) -> tuple[int, bool]:
  """從 Streamlit Plotly 選擇事件中解析出對應的資料索引"""
  if total_len <= 0:
    return 0, False
  default_idx = total_len - 1
  if not event or not getattr(event, "selection", None):
    return default_idx, False

  points = getattr(event.selection, "points", None) or []
  if not points:
    return default_idx, False

  pt = points[0] if isinstance(points[0], dict) else {}
  pt_idx = pt.get("point_index")
  if isinstance(pt_idx, int) and 0 <= pt_idx < total_len:
    return pt_idx, True

  x_val = pt.get("x")
  if x_val is not None:
    x_str = str(x_val)
    if x_str in labels:
      return labels.index(x_str), True

  return default_idx, False


def evaluate_intraday_alert(
    ticker_symbol: str,
    company_name: str,
    intra_df: pd.DataFrame,
    threshold_pct: float,
    est_high_2sd: float | None,
    est_low_2sd: float | None,
    has_bollinger: bool,
) -> dict:
  """彙整當日振幅與最新分時價對歷史布林通道（+2SD / -2SD）的突破/跌破狀態"""
  label = f"{company_name}（{ticker_symbol}）"
  if intra_df is None or intra_df.empty:
    return {
        "has_data": False,
        "has_alert": False,
        "amp_exceeded": False,
        "broke_upper": False,
        "broke_lower": False,
        "amplitude_pct": 0.0,
        "headline": f"目前無法取得 {label} 分時資料。",
        "details": [],
    }

  day_high = float(intra_df["High"].max())
  day_low = float(intra_df["Low"].min())
  high_time = intra_df["High"].idxmax().strftime("%H:%M")
  low_time = intra_df["Low"].idxmin().strftime("%H:%M")
  latest_price = float(intra_df["Close"].iloc[-1])

  amplitude_pts = day_high - day_low
  amplitude_pct = (amplitude_pts / day_low * 100) if day_low != 0 else 0.0
  amp_exceeded = amplitude_pct >= threshold_pct

  dist_to_high_pct = ((day_high - latest_price) / day_high * 100) if day_high != 0 else 0.0
  dist_from_low_pct = ((latest_price - day_low) / day_low * 100) if day_low != 0 else 0.0
  range_pos_pct = (
      ((latest_price - day_low) / amplitude_pts * 100)
      if amplitude_pts > 0
      else 50.0
  )

  pos_notes = []
  if dist_to_high_pct <= 0.25:
    pos_notes.append(f"接近／觸及當日高點 {day_high:.2f}（{high_time}）")
  if dist_from_low_pct <= 0.25:
    pos_notes.append(f"接近／觸及當日低點 {day_low:.2f}（{low_time}）")
  pos_desc = "、".join(pos_notes) if pos_notes else f"位於當日高低區間約 {range_pos_pct:.0f}% 位置"

  broke_upper = False
  broke_lower = False
  details = []

  if amp_exceeded:
    details.append(
        f"當日振幅 **{amplitude_pct:.2f}%**（高低點 **{day_low:.2f} ~ {day_high:.2f}**，價差 **{amplitude_pts:.2f}**）已達／超過設定門檻 **{threshold_pct:.2f}%**。"
    )
  else:
    details.append(
        f"當日振幅 **{amplitude_pct:.2f}%**（高低點 **{day_low:.2f} ~ {day_high:.2f}**，價差 **{amplitude_pts:.2f}**），低於門檻 **{threshold_pct:.2f}%**（尚差 {threshold_pct - amplitude_pct:.2f}%）。"
    )

  if has_bollinger and est_high_2sd is not None and est_low_2sd is not None:
    if latest_price > est_high_2sd:
      broke_upper = True
      diff_pts = latest_price - est_high_2sd
      diff_pct = (diff_pts / est_high_2sd * 100) if est_high_2sd != 0 else 0.0
      details.append(
          f"📈 **突破歷史布林通道預估高點 (+2SD)**：最新分時價 **{latest_price:.2f}** 高於預估壓力高點 **{est_high_2sd:.2f}**（超出 **+{diff_pts:.2f}** / **+{diff_pct:.2f}%**）。"
      )
    elif latest_price < est_low_2sd:
      broke_lower = True
      diff_pts = est_low_2sd - latest_price
      diff_pct = (diff_pts / est_low_2sd * 100) if est_low_2sd != 0 else 0.0
      details.append(
          f"📉 **跌破歷史布林通道預估低點 (-2SD)**：最新分時價 **{latest_price:.2f}** 低於預估支撐低點 **{est_low_2sd:.2f}**（跌破 **-{diff_pts:.2f}** / **-{diff_pct:.2f}%**）。"
      )
    else:
      up_gap_pct = ((est_high_2sd - latest_price) / latest_price * 100) if latest_price != 0 else 0.0
      dn_gap_pct = ((latest_price - est_low_2sd) / latest_price * 100) if latest_price != 0 else 0.0
      details.append(
          f"在歷史布林通道區間內（支撐 **-2SD {est_low_2sd:.2f}** ~ 壓力 **+2SD {est_high_2sd:.2f}**）：距上界尚有 **{up_gap_pct:.2f}%**，距下界尚有 **{dn_gap_pct:.2f}%**。"
      )
  else:
    details.append("歷史交易日數不足設定的 Window，暫無完整布林通道（+2SD / -2SD）可供比對。")

  details.append(f"最新分時價 **{latest_price:.2f}**（{pos_desc}）。")

  has_alert = amp_exceeded or broke_upper or broke_lower
  alert_tags = []
  if amp_exceeded:
    alert_tags.append("振幅超標")
  if broke_upper:
    alert_tags.append("突破 +2SD")
  if broke_lower:
    alert_tags.append("跌破 -2SD")

  headline = (
      f"🚨 **{label} 盤中提醒（{'／'.join(alert_tags)}）**"
      if has_alert
      else f"🔔 **{label} 盤中波動狀態正常**"
  )

  return {
      "has_data": True,
      "has_alert": has_alert,
      "amp_exceeded": amp_exceeded,
      "broke_upper": broke_upper,
      "broke_lower": broke_lower,
      "amplitude_pct": amplitude_pct,
      "headline": headline,
      "details": details,
  }


def render_bollinger_chart(
    ticker_symbol: str,
    company_name: str,
    calc_df: pd.DataFrame,
    window: int,
    end_label: str,
):
  """繪製可互動式歷史股價與布林通道高低點預估圖，並在下方即時顯示點選日期精準數據"""
  fig = go.Figure()
  dates = calc_df.index.strftime("%Y-%m-%d").tolist()

  fig.add_trace(
      go.Scatter(
          x=dates,
          y=calc_df["Close"],
          mode="lines+markers",
          name="Close Price",
          marker=dict(size=4),
          line=dict(color="#111827", width=2),
          hovertemplate="日期：%{x}<br>收盤價：%{y:.2f}<extra></extra>",
      )
  )

  if {"SMA", "Estimated_High", "Estimated_Low"}.issubset(calc_df.columns):
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=calc_df["Estimated_High"],
            mode="lines",
            name="Estimated High (+2SD)",
            line=dict(color="#dc2626", width=1.4, dash="dash"),
            hovertemplate="日期：%{x}<br>預估高點 (+2SD)：%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=calc_df["Estimated_Low"],
            mode="lines",
            name="Estimated Low (-2SD)",
            line=dict(color="#16a34a", width=1.4, dash="dash"),
            fill="tonexty",
            fillcolor="rgba(148, 163, 184, 0.16)",
            hovertemplate="日期：%{x}<br>預估低點 (-2SD)：%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=calc_df["SMA"],
            mode="lines",
            name=f"{window}-Day SMA",
            line=dict(color="#2563eb", width=1.6, dash="dot"),
            hovertemplate="日期：%{x}<br>基準均價 SMA：%{y:.2f}<extra></extra>",
        )
    )

  fig.update_layout(
      title=f"{company_name} ({ticker_symbol}) 歷史極值與布林通道（{end_label}）",
      xaxis_title="日期",
      yaxis_title="價格",
      hovermode="x unified",
      clickmode="event+select",
      height=460,
      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
      margin=dict(t=80, b=40, l=40, r=20),
  )

  chart_event = st.plotly_chart(
      fig,
      use_container_width=True,
      key=f"bb_chart_{ticker_symbol}",
      on_select="rerun",
      selection_mode="points",
  )

  idx, is_clicked = _resolve_selected_index(chart_event, dates, len(calc_df))
  row = calc_df.iloc[idx]
  sel_date = dates[idx]

  st.markdown(f"##### 📍 布林通道圖點選日期精準數據（`{sel_date}`）")
  st.caption(
      "目前顯示**已點選日期**數據" if is_clicked else "點擊上方布林通道圖任一日期，即可查看該日精準數值（預設顯示區間最新一日）。"
  )

  c1, c2, c3, c4 = st.columns(4)
  c1.metric("收盤價", f"{float(row['Close']):.2f}")
  c2.metric("當日最高 / 最低", f"{float(row['High']):.2f} / {float(row['Low']):.2f}")
  if "SMA" in row and pd.notna(row["SMA"]):
    c3.metric(f"基準均價 (SMA {window}日)", f"{float(row['SMA']):.2f}")
  else:
    c3.metric(f"基準均價 (SMA {window}日)", "資料不足")

  if "Estimated_High" in row and "Estimated_Low" in row and pd.notna(row["Estimated_High"]) and pd.notna(row["Estimated_Low"]):
    c4.metric(
        "預估通道 (+2SD / -2SD)",
        f"{float(row['Estimated_High']):.2f} / {float(row['Estimated_Low']):.2f}",
    )
  else:
    c4.metric("預估通道 (+2SD / -2SD)", "資料不足")


def render_intraday_volatility(
    ticker_symbol: str,
    company_name: str,
    intra_df: pd.DataFrame,
    source_desc: str,
    threshold_pct: float,
    est_high_2sd: float | None,
    est_low_2sd: float | None,
    has_bollinger: bool,
):
  """繪製可互動式當日波動圖並顯示盤中波動提醒與點選時刻精準數據"""
  st.markdown(f"#### ⚡ {company_name}（{ticker_symbol}）當日波動圖與盤中提醒（{source_desc}）")

  if intra_df is None or intra_df.empty:
    st.info("目前無法取得該股票的分時波動資料（可能處於休市或資料源暫時無回應）。")
    return

  alert_info = evaluate_intraday_alert(
      ticker_symbol=ticker_symbol,
      company_name=company_name,
      intra_df=intra_df,
      threshold_pct=threshold_pct,
      est_high_2sd=est_high_2sd,
      est_low_2sd=est_low_2sd,
      has_bollinger=has_bollinger,
  )

  banner_body = f"{alert_info['headline']}\n\n" + "\n".join(
      f"- {line}" for line in alert_info["details"]
  )
  if alert_info["has_alert"]:
    st.warning(banner_body)
  else:
    st.info(banner_body)

  work = intra_df.copy()
  day_high = float(work["High"].max())
  day_low = float(work["Low"].min())
  high_time = work["High"].idxmax().strftime("%H:%M")
  low_time = work["Low"].idxmin().strftime("%H:%M")
  latest_price = float(work["Close"].iloc[-1])
  open_price = (
      float(work["Open"].iloc[0])
      if "Open" in work.columns and pd.notna(work["Open"].iloc[0])
      else float(work["Close"].iloc[0])
  )

  amplitude_pts = day_high - day_low
  amplitude_pct = alert_info["amplitude_pct"]
  change_pct = ((latest_price - open_price) / open_price * 100) if open_price != 0 else 0.0

  if has_bollinger and est_high_2sd is not None and est_low_2sd is not None:
    if alert_info["broke_upper"]:
      bb_status_label = "突破 +2SD"
      bb_delta = f"最新 {latest_price:.2f} > 高界 {est_high_2sd:.2f}"
    elif alert_info["broke_lower"]:
      bb_status_label = "跌破 -2SD"
      bb_delta = f"最新 {latest_price:.2f} < 低界 {est_low_2sd:.2f}"
    else:
      bb_status_label = "通道區間內"
      bb_delta = f"-2SD {est_low_2sd:.2f} ~ +2SD {est_high_2sd:.2f}"
  else:
    bb_status_label = "資料不足"
    bb_delta = "需達 Window 天數"

  m1, m2, m3, m4, m5 = st.columns(5)
  m1.metric("最新分時價", f"{latest_price:.2f}", f"相對首筆 {change_pct:+.2f}%")
  m2.metric("當日最高價", f"{day_high:.2f}", f"出現於 {high_time}")
  m3.metric("當日最低價", f"{day_low:.2f}", f"出現於 {low_time}")
  m4.metric(
      "當日高低振幅",
      f"{amplitude_pct:.2f}%",
      f"價差 {amplitude_pts:.2f}（門檻 {threshold_pct:.1f}%）",
  )
  m5.metric("布林通道狀態", bb_status_label, bb_delta)

  typical_price = (work["High"] + work["Low"] + work["Close"]) / 3.0
  has_volume = "Volume" in work.columns and work["Volume"].fillna(0).sum() > 0
  if has_volume:
    vol = work["Volume"].fillna(0)
    cum_vol = vol.cumsum().replace(0, np.nan)
    vwap = (typical_price * vol).cumsum() / cum_vol
    vwap_label = "Intraday VWAP"
  else:
    vwap = typical_price.expanding().mean()
    vwap_label = "Avg Typical Price"

  work["Calc_VWAP"] = vwap
  time_labels = work.index.strftime("%Y-%m-%d %H:%M").tolist()

  if has_volume:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.74, 0.26],
    )
  else:
    fig = make_subplots(rows=1, cols=1)

  fig.add_trace(
      go.Scatter(
          x=time_labels,
          y=work["High"],
          mode="lines",
          name="分時最高",
          line=dict(color="rgba(59, 130, 246, 0.35)", width=1),
          hovertemplate="時間：%{x}<br>分時最高：%{y:.2f}<extra></extra>",
      ),
      row=1,
      col=1,
  )
  fig.add_trace(
      go.Scatter(
          x=time_labels,
          y=work["Low"],
          mode="lines",
          name="分時最低",
          line=dict(color="rgba(59, 130, 246, 0.35)", width=1),
          fill="tonexty",
          fillcolor="rgba(59, 130, 246, 0.14)",
          hovertemplate="時間：%{x}<br>分時最低：%{y:.2f}<extra></extra>",
      ),
      row=1,
      col=1,
  )
  fig.add_trace(
      go.Scatter(
          x=time_labels,
          y=work["Close"],
          mode="lines+markers",
          name="分時收盤",
          marker=dict(size=5),
          line=dict(color="#111827", width=2),
          hovertemplate="時間：%{x}<br>分時價格：%{y:.2f}<extra></extra>",
      ),
      row=1,
      col=1,
  )
  fig.add_trace(
      go.Scatter(
          x=time_labels,
          y=work["Calc_VWAP"],
          mode="lines",
          name=vwap_label,
          line=dict(color="#d97706", width=1.6, dash="dash"),
          hovertemplate=f"時間：%{{x}}<br>{vwap_label}：%{{y:.2f}}<extra></extra>",
      ),
      row=1,
      col=1,
  )

  fig.add_hline(
      y=day_high,
      line_dash="dot",
      line_color="#dc2626",
      annotation_text=f"當日高 {day_high:.2f}",
      annotation_position="top left",
      row=1,
      col=1,
  )
  fig.add_hline(
      y=day_low,
      line_dash="dot",
      line_color="#16a34a",
      annotation_text=f"當日低 {day_low:.2f}",
      annotation_position="bottom left",
      row=1,
      col=1,
  )

  if has_bollinger and est_high_2sd is not None and est_low_2sd is not None:
    fig.add_hline(
        y=est_high_2sd,
        line_dash="dashdot",
        line_color="#991b1b",
        annotation_text=f"歷史 +2SD {est_high_2sd:.2f}",
        annotation_position="top right",
        row=1,
        col=1,
    )
    fig.add_hline(
        y=est_low_2sd,
        line_dash="dashdot",
        line_color="#166534",
        annotation_text=f"歷史 -2SD {est_low_2sd:.2f}",
        annotation_position="bottom right",
        row=1,
        col=1,
    )

  if has_volume:
    open_series = work["Open"] if "Open" in work.columns else work["Close"].shift(1).fillna(work["Close"])
    bar_colors = [
        "#ef4444" if c >= o else "#22c55e"
        for c, o in zip(work["Close"], open_series)
    ]
    fig.add_trace(
        go.Bar(
            x=time_labels,
            y=work["Volume"].fillna(0),
            name="成交量",
            marker_color=bar_colors,
            opacity=0.7,
            hovertemplate="時間：%{x}<br>成交量：%{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="成交量", row=2, col=1)

  fig.update_yaxes(title_text="價格", row=1, col=1)
  fig.update_layout(
      title=f"{company_name} ({ticker_symbol}) 互動式當日波動圖（{source_desc}）",
      hovermode="x unified",
      clickmode="event+select",
      height=540 if has_volume else 440,
      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
      margin=dict(t=80, b=40, l=40, r=20),
  )

  intra_event = st.plotly_chart(
      fig,
      use_container_width=True,
      key=f"intra_chart_{ticker_symbol}",
      on_select="rerun",
      selection_mode="points",
  )

  idx, is_clicked = _resolve_selected_index(intra_event, time_labels, len(work))
  sel_row = work.iloc[idx]
  sel_time = time_labels[idx]
  sel_open = float(sel_row["Open"]) if "Open" in sel_row and pd.notna(sel_row["Open"]) else float(sel_row["Close"])
  sel_high = float(sel_row["High"])
  sel_low = float(sel_row["Low"])
  sel_close = float(sel_row["Close"])
  sel_vwap = float(sel_row["Calc_VWAP"]) if pd.notna(sel_row["Calc_VWAP"]) else sel_close
  sel_vol = float(sel_row["Volume"]) if "Volume" in sel_row and pd.notna(sel_row["Volume"]) else None

  st.markdown(f"##### 📍 當日波動圖點選時刻精準數據（`{sel_time}`）")
  st.caption(
      "目前顯示**已點選時刻**數據" if is_clicked else "點擊上方當日波動圖任一時間點，即可查看該時刻精準數值（預設顯示最新一筆）。"
  )

  p1, p2, p3, p4, p5 = st.columns(5)
  p1.metric("該時刻收盤價", f"{sel_close:.2f}", f"開盤 {sel_open:.2f}")
  p2.metric("該時刻最高價", f"{sel_high:.2f}")
  p3.metric("該時刻最低價", f"{sel_low:.2f}")
  p4.metric(vwap_label, f"{sel_vwap:.2f}", f"價差 {sel_close - sel_vwap:+.2f}")
  if sel_vol is not None and sel_vol > 0:
    p5.metric("該時刻成交量", f"{sel_vol:,.0f}")
  else:
    p5.metric("該時刻成交量", "無成交量資料")


def render_ai_buy_point_analysis(
    ticker_symbol: str,
    company_name: str,
    summary: dict,
    intra_df: pd.DataFrame,
    fund_raw: dict,
    threshold_pct: float,
):
  """依據技術面通道位置、當日波動與基本面指標進行 AI 量化買點評估"""
  st.markdown(f"#### 🤖 AI 當前購買點評估 — {company_name}（{ticker_symbol}）")

  score = 50.0
  signals = []
  buy_zones = []

  curr_price = summary.get("current_price_val")
  sma_val = summary.get("latest_sma_val")
  high_2sd = summary.get("latest_est_high_val")
  low_2sd = summary.get("latest_est_low_val")
  has_bb = bool(summary.get("has_enough_window"))

  intra_latest = None
  intra_vwap_latest = None
  intra_low = None
  intra_high = None
  amp_pct = None

  if intra_df is not None and not intra_df.empty:
    intra_latest = float(intra_df["Close"].iloc[-1])
    intra_low = float(intra_df["Low"].min())
    intra_high = float(intra_df["High"].max())
    amp_pct = ((intra_high - intra_low) / intra_low * 100) if intra_low != 0 else 0.0
    tp = (intra_df["High"] + intra_df["Low"] + intra_df["Close"]) / 3.0
    if "Volume" in intra_df.columns and intra_df["Volume"].fillna(0).sum() > 0:
      vol = intra_df["Volume"].fillna(0).cumsum().replace(0, np.nan)
      vwap_series = (tp * intra_df["Volume"].fillna(0)).cumsum() / vol
    else:
      vwap_series = tp.expanding().mean()
    intra_vwap_latest = float(vwap_series.iloc[-1])

  ref_price = intra_latest if intra_latest is not None else curr_price

  # 1. 歷史布林通道位置評估
  if has_bb and high_2sd is not None and low_2sd is not None and sma_val is not None and ref_price is not None:
    span = high_2sd - low_2sd
    if span > 0:
      pos_ratio = (ref_price - low_2sd) / span
    else:
      pos_ratio = 0.5

    if ref_price <= low_2sd:
      score += 22
      signals.append(f"價格 **({ref_price:.2f})** 已觸及或跌破歷史布林下界 **-2SD ({low_2sd:.2f})**，統計上屬超跌支撐區，具備逢低分批試單條件。")
    elif pos_ratio <= 0.30:
      score += 14
      signals.append(f"價格 **({ref_price:.2f})** 位於布林通道偏下緣（區間分位 {pos_ratio*100:.0f}%），靠近支撐 **-2SD ({low_2sd:.2f})**，風險報酬比相對較佳。")
    elif ref_price >= high_2sd:
      score -= 22
      signals.append(f"價格 **({ref_price:.2f})** 已超越歷史布林上界 **+2SD ({high_2sd:.2f})**，短線處於統計高檔延伸區，追高回檔壓力較大。")
    elif pos_ratio >= 0.75:
      score -= 12
      signals.append(f"價格 **({ref_price:.2f})** 靠近上方壓力 **+2SD ({high_2sd:.2f})**（區間分位 {pos_ratio*100:.0f}%），建議等待拉回均線或支撐再考慮。")
    else:
      score += 4
      signals.append(f"價格 **({ref_price:.2f})** 位於布林通道中段（基準均價 SMA **{sma_val:.2f}** 附近），趨勢偏震盪整理。")

    buy_zones.append(f"布林支撐參考區：**{low_2sd:.2f} ~ {sma_val:.2f}**（保守分批留意 **{low_2sd:.2f}** 附近）")

  # 2. 當日分時與振幅評估
  if intra_latest is not None and intra_vwap_latest is not None and intra_low is not None and intra_high is not None:
    if intra_latest <= intra_vwap_latest:
      score += 7
      signals.append(f"最新分時價 **({intra_latest:.2f})** 位於當日均價／VWAP **({intra_vwap_latest:.2f})** 下方，有利觀察承接力道。")
    else:
      score -= 4
      signals.append(f"最新分時價 **({intra_latest:.2f})** 站上當日均價／VWAP **({intra_vwap_latest:.2f})**，短線多方較強但需留意乖離。")

    if amp_pct is not None and amp_pct >= threshold_pct:
      score -= 8
      signals.append(f"當日振幅已達 **{amp_pct:.2f}%**（門檻 {threshold_pct:.1f}%），盤中震盪劇烈，建議縮小單筆部位或分批進場。")
    elif amp_pct is not None:
      score += 3
      signals.append(f"當日振幅為 **{amp_pct:.2f}%**（低於門檻 {threshold_pct:.1f}%），盤中波動相對溫和。")

    buy_zones.append(f"當日分時觀察區：**{intra_low:.2f} ~ {intra_vwap_latest:.2f}**（防守可參考當日低點 **{intra_low:.2f}** 下方）")

  # 3. 基本面體質評估
  pe = fund_raw.get("pe")
  pm = fund_raw.get("profit_margin_pct")
  debt = fund_raw.get("debt_ratio_pct")

  if pm is not None and pm >= 15:
    score += 5
    signals.append(f"淨利率達 **{pm:.2f}%**，獲利能力穩健。")
  elif pm is not None and pm < 0:
    score -= 10
    signals.append(f"目前淨利率為負（**{pm:.2f}%**），基本面獲利仍待改善。")

  if debt is not None and debt <= 60:
    score += 4
    signals.append(f"負債率約 **{debt:.2f}%**，財務結構相對穩健。")
  elif debt is not None and debt >= 80:
    score -= 5
    signals.append(f"負債率偏高（約 **{debt:.2f}%**），需留意利息與財務槓桿風險。")

  if pe is not None and 0 < pe <= 20:
    score += 4
    signals.append(f"本益比約 **{pe:.2f}**，估值相對合理。")
  elif pe is not None and pe > 45:
    score -= 4
    signals.append(f"本益比約 **{pe:.2f}**，市場已反映較高成長預期。")

  score = max(0.0, min(100.0, score))

  if score >= 68:
    verdict = "🟢 偏多／逢低分批買點區"
    badge_fn = st.success
    summary_adv = "目前價格位置與統計支撐／基本面條件較具吸引力，可採「分批進場、嚴設停損」方式布局。"
  elif score >= 45:
    verdict = "🟡 中性／觀察等待確認買點"
    badge_fn = st.info
    summary_adv = "目前訊號多空交織或處於中性區間，建議先觀察支撐是否守穩或等待拉回關鍵價位再分批出手。"
  else:
    verdict = "🔴 偏高或震盪風險／不建議追高"
    badge_fn = st.warning
    summary_adv = "目前位置偏高或盤中波動較大，建議避免貿然追價，耐心等待回測支撐或波動收斂。"

  badge_fn(
      f"**AI 買點評估結論：{verdict}（綜合分數：{score:.0f} / 100）**\n\n{summary_adv}"
  )

  if buy_zones:
    st.markdown("**🎯 參考買點與風控區間：**")
    for bz in buy_zones:
      st.markdown(f"- {bz}")

  st.markdown("**📌 評估依據拆解：**")
  for sig in signals:
    st.markdown(f"- {sig}")

  st.caption("⚠️ 以上為系統依歷史統計、分時數據與公開財報指標產生的量化分析參考，不構成投資買賣建議，請依自身風險承受度審慎決策。")


def run_full_analysis():
  """執行完整分析並存入 session_state"""
  end_date_str = (
      datetime.today().strftime("%Y-%m-%d")
      if use_today
      else custom_end_date.strftime("%Y-%m-%d")
  )
  start_date_str = start_date.strftime("%Y-%m-%d")
  tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

  if not tickers:
    st.session_state["analysis_bundle"] = None
    st.error("❌ 請至少輸入一個股票代號（例如 2330.TW 或 SPY）。")
    return

  index_labels = [
      "公司性質",
      "每股盈餘 (EPS)",
      "本益比 (P/E)",
      "毛利率",
      "淨利率",
      "流動資產 (最新期)",
      "負債率 (總負債/總資產)",
  ]
  combined_fundamentals = {}
  per_ticker_results = {}

  progress = st.progress(0, text="準備抓取資料...")
  total_steps = len(tickers)

  for idx, t in enumerate(tickers):
    progress.progress(
        idx / total_steps,
        text=f"⏳ 正在分析 {t} 基本面、歷史區間與當日波動（{idx + 1}/{total_steps}）...",
    )

    fund_info = get_fundamental_info(t)
    company_name = fund_info["company_name"] if fund_info else t
    fund_raw = fund_info["raw"] if fund_info else {}
    if fund_info:
      combined_fundamentals[fund_info["col_title"]] = fund_info["metrics"]

    hist_df = download_history_with_retry(t, start_date_str, end_date_str)
    intra_df, intra_desc = download_intraday_data(t, interval=intraday_interval)

    if hist_df.empty:
      per_ticker_results[t] = {
          "status": "empty",
          "company_name": company_name,
          "fund_raw": fund_raw,
          "message": f"❌ {company_name}（{t}）在 {start_date_str} ~ {end_date_str} 期間無法取得歷史股價資料。",
          "intra_df": intra_df,
          "intra_desc": intra_desc,
      }
    else:
      summary, calc_df = compute_analysis(hist_df, window=int(window))
      per_ticker_results[t] = {
          "status": "ok",
          "company_name": company_name,
          "fund_raw": fund_raw,
          "summary": summary,
          "calc_df": calc_df,
          "intra_df": intra_df,
          "intra_desc": intra_desc,
      }

    if idx < len(tickers) - 1:
      time.sleep(0.5)

  progress.progress(1.0, text="✅ 分析完成")
  time.sleep(0.2)
  progress.empty()

  fund_df = (
      pd.DataFrame(combined_fundamentals, index=index_labels)
      if combined_fundamentals
      else pd.DataFrame()
  )
  if not fund_df.empty:
    fund_df.index.name = "財務指標"

  st.session_state["analysis_bundle"] = {
      "tickers": tickers,
      "start_date_str": start_date_str,
      "end_date_str": end_date_str,
      "window": int(window),
      "intraday_interval": intraday_interval,
      "fund_df": fund_df,
      "per_ticker_results": per_ticker_results,
      "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
  }


def refresh_intraday_only():
  """僅重新抓取各檔股票的當日波動資料，保留既有基本面與歷史分析結果"""
  bundle = st.session_state.get("analysis_bundle")
  if not bundle:
    return

  download_intraday_data.clear()
  tickers = bundle["tickers"]
  iv = bundle.get("intraday_interval", intraday_interval)
  per_ticker_results = bundle["per_ticker_results"]

  for t in tickers:
    intra_df, intra_desc = download_intraday_data(t, interval=iv)
    if t in per_ticker_results:
      per_ticker_results[t]["intra_df"] = intra_df
      per_ticker_results[t]["intra_desc"] = intra_desc

  bundle["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  st.session_state["analysis_bundle"] = bundle


# --- 觸發完整分析或自動刷新 ---
if run_btn:
  st.session_state["last_refresh_ts"] = time.time()
  run_full_analysis()
elif auto_refresh and st.session_state.get("analysis_bundle") is None:
  st.session_state["last_refresh_ts"] = time.time()
  run_full_analysis()


bundle = st.session_state.get("analysis_bundle")

if bundle:
  tickers = bundle["tickers"]
  start_date_str = bundle["start_date_str"]
  end_date_str = bundle["end_date_str"]
  win = bundle["window"]
  fund_df = bundle["fund_df"]
  per_ticker_results = bundle["per_ticker_results"]
  last_updated = bundle.get("last_updated", "")

  status_text = (
      f"🕒 最後更新時間：**{last_updated}**"
      f" ｜ 🎯 振幅提醒門檻：**{float(amplitude_threshold):.1f}%**"
      " ｜ 📐 比對基準：**布林通道 ±2SD**"
      + (
          f" ｜ 🟢 自動刷新已啟用（每 {int(refresh_interval)} 秒更新當日波動圖）"
          if auto_refresh
          else " ｜ ⚪ 自動刷新已關閉"
      )
  )
  st.caption(status_text)

  # 1. 全檔基本面比較總表
  if not fund_df.empty:
    st.subheader("📋 股票基本面比較總表")
    st.dataframe(fund_df, use_container_width=True)
    st.download_button(
        "📥 下載基本面比較總表 (CSV)",
        data=fund_df.to_csv().encode("utf-8-sig"),
        file_name=f"股票基本面比較表_{end_date_str}.csv",
        mime="text/csv",
        key="download_all_fundamentals",
    )

  st.divider()
  st.subheader("🗂️ 多檔股票分頁詳情、互動波動圖與 AI 買點分析")

  # 2. 多檔股票 Tab 分頁切換（分頁標題同時顯示公司名稱與代號）
  tab_titles = []
  for t in tickers:
    res = per_ticker_results.get(t, {})
    comp_name = res.get("company_name", t)
    summary = res.get("summary", {})
    alert_eval = evaluate_intraday_alert(
        ticker_symbol=t,
        company_name=comp_name,
        intra_df=res.get("intra_df", pd.DataFrame()),
        threshold_pct=float(amplitude_threshold),
        est_high_2sd=summary.get("latest_est_high_val"),
        est_low_2sd=summary.get("latest_est_low_val"),
        has_bollinger=bool(summary.get("has_enough_window", False)),
    )
    alert_prefix = "🚨 " if alert_eval["has_alert"] else ""
    tab_titles.append(f"{alert_prefix}📌 {comp_name} ({t})")

  tabs = st.tabs(tab_titles)

  for tab, t in zip(tabs, tickers):
    with tab:
      res = per_ticker_results.get(t)
      if not res:
        st.warning(f"❌ 找不到 {t} 的分析資料。")
        continue

      company_name = res.get("company_name", t)
      fund_raw = res.get("fund_raw", {})
      summary = res.get("summary", {})

      # 區塊 0：優先顯現公司名稱與股票代號
      st.markdown(f"## 🏢 {company_name}（`{t}`）")

      # 區塊 A：可互動當日波動圖、盤中波動提醒與點選時刻精準數據
      render_intraday_volatility(
          ticker_symbol=t,
          company_name=company_name,
          intra_df=res.get("intra_df", pd.DataFrame()),
          source_desc=res.get("intra_desc", "無分時資料"),
          threshold_pct=float(amplitude_threshold),
          est_high_2sd=summary.get("latest_est_high_val"),
          est_low_2sd=summary.get("latest_est_low_val"),
          has_bollinger=bool(summary.get("has_enough_window", False)),
      )

      st.divider()

      # 區塊 B：歷史極值與可互動布林通道高低點預估 + 點選日期精準數據
      if res["status"] != "ok":
        st.warning(res["message"])
        continue

      calc_df = res["calc_df"]

      st.markdown(
          f"#### 📉 {company_name}（{t}）歷史極值與布林通道（{start_date_str} ~ {end_date_str}）"
      )

      r1_c1, r1_c2, r1_c3 = st.columns(3)
      r1_c1.metric("最新收盤價", summary["最新收盤價"])
      r1_c2.metric("區間最高價", summary["區間最高價"], f"發生於 {summary['區間最高價日期']}")
      r1_c3.metric("區間最低價", summary["區間最低價"], f"發生於 {summary['區間最低價日期']}")

      if summary["has_enough_window"]:
        r2_c1, r2_c2, r2_c3 = st.columns(3)
        r2_c1.metric(f"基準均價 (SMA {win}日)", summary[f"基準均價 (SMA {win}日)"])
        r2_c2.metric("預估壓力高點 (+2SD)", summary["預估壓力高點 (+2SD)"])
        r2_c3.metric("預估支撐低點 (-2SD)", summary["預估支撐低點 (-2SD)"])
      else:
        st.info(
            f"⚠️ 有效交易日（{len(calc_df)} 天）少於設定的 window={win} 天，"
            "目前僅顯示歷史極值，尚未產生完整布林通道上下界。"
        )

      render_bollinger_chart(
          ticker_symbol=t,
          company_name=company_name,
          calc_df=calc_df,
          window=win,
          end_label=end_date_str,
      )

      # 區塊 C：單檔獨立匯出
      internal_keys = {
          "has_enough_window",
          "current_price_val",
          "latest_sma_val",
          "latest_est_high_val",
          "latest_est_low_val",
      }
      summary_rows = [{"項目": "公司名稱", "數值": company_name}, {"項目": "股票代號", "數值": t}] + [
          {"項目": k, "數值": v}
          for k, v in summary.items()
          if k not in internal_keys
      ]
      summary_df = pd.DataFrame(summary_rows)

      history_export_df = calc_df.copy()
      history_export_df.index = history_export_df.index.strftime("%Y-%m-%d")
      history_export_df.index.name = "Date"

      dl_col1, dl_col2 = st.columns(2)
      with dl_col1:
        st.download_button(
            f"📄 匯出 {company_name} ({t}) 分析摘要 (CSV)",
            data=summary_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{t}_分析摘要_{start_date_str}_{end_date_str}.csv",
            mime="text/csv",
            key=f"dl_summary_{t}",
            use_container_width=True,
        )
      with dl_col2:
        st.download_button(
            f"📊 匯出 {company_name} ({t}) 歷史股價與指標 (CSV)",
            data=history_export_df.to_csv().encode("utf-8-sig"),
            file_name=f"{t}_歷史股價與布林通道_{start_date_str}_{end_date_str}.csv",
            mime="text/csv",
            key=f"dl_hist_{t}",
            use_container_width=True,
        )

      with st.expander(f"檢視 {company_name} ({t}) 歷史股價與指標明細"):
        st.dataframe(history_export_df.tail(30), use_container_width=True)

      st.divider()

      # 區塊 D：最後由 AI 評估當前適合購買點
      render_ai_buy_point_analysis(
          ticker_symbol=t,
          company_name=company_name,
          summary=summary,
          intra_df=res.get("intra_df", pd.DataFrame()),
          fund_raw=fund_raw,
          threshold_pct=float(amplitude_threshold),
      )

  # --- 自動刷新計時器 ---
  if auto_refresh:
    last_ts = st.session_state.get("last_refresh_ts", time.time())
    elapsed = time.time() - last_ts
    remaining = max(0, int(refresh_interval - elapsed))
    countdown_slot = st.sidebar.empty()
    countdown_slot.caption(f"⏳ 下次更新當日波動圖約剩 {remaining} 秒")

    time.sleep(1)
    if time.time() - last_ts >= int(refresh_interval):
      st.session_state["last_refresh_ts"] = time.time()
      refresh_intraday_only()
    st.rerun()
else:
  st.info("👈 請在左側設定股票代號與參數後，點擊「開始分析」。")
