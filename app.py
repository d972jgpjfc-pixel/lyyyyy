from datetime import datetime
import re
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


def _format_large_num(val, currency: str = "", is_pct: bool = False) -> str:
  """將大數字或百分比格式化為易讀字串"""
  if val is None or not isinstance(val, (int, float)) or pd.isna(val):
    return "N/A"
  if is_pct:
    return f"{float(val) * 100:.2f}%"
  num = float(val)
  abs_num = abs(num)
  unit_suffix = f" ({currency})" if currency else ""
  if abs_num >= 1e12:
    return f"{num / 1e12:,.2f} 兆{unit_suffix}"
  if abs_num >= 1e8:
    return f"{num / 1e8:,.2f} 億{unit_suffix}"
  if abs_num >= 1e4:
    return f"{num / 1e4:,.2f} 萬{unit_suffix}"
  return f"{num:,.2f}{unit_suffix}"


def extract_business_and_rd(info: dict, stock_obj: yf.Ticker) -> tuple[str, str, list[str]]:
  """從公司簡介、產業分類與最新動態中，萃取「主要業務」、「未來研發方向」與「近期新聞摘要」"""
  quote_type = info.get("quoteType", "")
  sector = info.get("sector", "")
  industry = info.get("industry", "")
  summary_text = info.get("longBusinessSummary", "") or ""
  summary_lower = summary_text.lower()

  recent_news = []
  try:
    news_list = stock_obj.news or []
    for item in news_list[:5]:
      content = item.get("content", {}) if isinstance(item, dict) else {}
      title = item.get("title") or content.get("title") or ""
      pub_date = (
          item.get("providerPublishTime")
          or content.get("pubDate")
          or ""
      )
      date_str = ""
      if isinstance(pub_date, (int, float)) and pub_date > 0:
        date_str = datetime.fromtimestamp(pub_date).strftime("%Y-%m-%d")
      elif isinstance(pub_date, str) and len(pub_date) >= 10:
        date_str = pub_date[:10]
      if title:
        recent_news.append(f"[{date_str}] {title}" if date_str else title)
  except Exception:
    pass

  # 1. 若為 ETF / 基金
  if quote_type == "ETF" or "etf" in (info.get("shortName", "") + info.get("longName", "")).lower():
    what_it_does = "指數股票型基金 (ETF)：追蹤特定大盤指數、產業或一籃子資產表現。"
    if summary_text:
      first_sent = summary_text.split(".")[0].strip()
      if len(first_sent) > 10:
        what_it_does += f"（{first_sent[:110]}...）" if len(first_sent) > 110 else f"（{first_sent}）"
    future_rd = "無實體產品研發（主要隨追蹤指數成分股權重調整與再平衡）。"
    return what_it_does, future_rd, recent_news

  # 2. 解析「主要業務（做什麼的）」
  business_tags = []
  biz_map = [
      (["foundry", "semiconductor", "integrated circuit", "wafer", "chips"], "半導體／積體電路製造與晶片設計"),
      (["gpu", "graphics", "accelerated computing", "data center"], "GPU 繪圖處理器、資料中心與高速運算架構"),
      (["software", "cloud", "saas", "platform", "operating system"], "企業軟體、雲端運算服務與數位平台"),
      (["smartphone", "consumer electronics", "iphone", "personal computer", "hardware"], "消費性電子產品、智慧型手機與硬體設備"),
      (["electric vehicle", "autonomous", "automotive", "battery", "energy storage"], "電動車製造、車用電子與儲能系統"),
      (["pharmaceutical", "biotech", "drug", "clinical", "therapeutics", "medical"], "新藥研發、生物製藥與醫療器材"),
      (["bank", "financial", "insurance", "asset management", "investment"], "金融銀行、保險與資產管理服務"),
      (["retail", "e-commerce", "consumer", "apparel", "food", "beverage"], "零售通路、電子商務與民生消費品"),
      (["steel", "chemical", "material", "petrochemical", "metal"], "鋼鐵、塑化與工業基礎原物料製造"),
      (["telecom", "wireless", "5g", "network", "communication"], "電信網路通訊與基礎建設服務"),
      (["shipping", "airline", "logistics", "transportation", "marine"], "海空航運、物流與交通運輸服務"),
  ]
  for kws, desc in biz_map:
    if any(k in summary_lower for k in kws):
      business_tags.append(desc)

  if business_tags:
    what_it_does = "、".join(dict.fromkeys(business_tags[:2]))
  elif industry and sector:
    what_it_does = f"{sector} 領域之 {industry} 相關產品與服務"
  elif summary_text:
    what_it_does = summary_text.split(".")[0].strip()[:90]
  else:
    what_it_does = "暫無詳細業務說明"

  # 3. 解析「未來研發方向（將研發什麼）」
  rd_tags = []
  rd_map = [
      (["artificial intelligence", "ai ", "generative ai", "machine learning", "deep learning"], "生成式 AI 算力架構、人工智慧模型與邊緣 AI 應用"),
      (["2nm", "3nm", "advanced node", "nanometer", "packaging", "cowos", "silicon photonics"], "次世代先進奈米製程、3D 先進封裝與矽光子技術"),
      (["autonomous", "self-driving", "robotics", "humanoid"], "全自動駕駛演算法與智慧機器人系統"),
      (["quantum", "next-generation", "6g", "satellite"], "次世代通訊（6G/衛星）與前瞻運算技術"),
      (["clean energy", "renewable", "solar", "hydrogen", "carbon", "ev "], "低碳綠能技術、高效能電池與永續能源方案"),
      (["clinical trial", "phase", "pipeline", "oncology", "gene", "immunotherapy"], "新適應症臨床試驗、精準醫療與創新療法管線"),
      (["cloud", "cybersecurity", "zero trust", "enterprise"], "次世代雲端原生架構與 AI 資安防護技術"),
  ]
  for kws, desc in rd_map:
    if any(k in summary_lower for k in kws):
      rd_tags.append(desc)

  news_hints = [
      title[:55]
      for title in recent_news
      if any(w in title.lower() for w in ["ai", "next-gen", "launch", "chip", "r&d", "develop", "battery", "trial", "2nm", "3nm", "invest"])
  ]

  rd_sentences = [
      s.strip()
      for s in re.split(r"[.。]", summary_text)
      if any(w in s.lower() for w in ["research", "develop", "next-generation", "future", "innovation", "pipeline"])
  ]

  if rd_tags:
    future_rd = "重點投入：" + "、".join(dict.fromkeys(rd_tags[:2]))
  elif rd_sentences:
    future_rd = f"聚焦產品技術升級與創新研發（{rd_sentences[0][:85]}...）"
  elif sector:
    future_rd = f"持續深化 {sector} 核心技術升級、AI 數位化導入與新一代產品線開發"
  else:
    future_rd = "持續投入核心產品升級與新技術應用開發"

  if news_hints:
    future_rd += f"｜近期動向：{news_hints[0]}"

  return what_it_does, future_rd, recent_news


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
  """抓取單一股票基本面指標、公司檔案、主要業務與未來研發方向"""
  try:
    stock = yf.Ticker(ticker_symbol)
    info = stock.info or {}
    bs = stock.balance_sheet

    if not info and (bs is None or bs.empty):
      return {
          "company_name": ticker_symbol,
          "what_it_does": "無資料",
          "future_rd": "無資料",
          "profile": {},
          "recent_news": [],
          "long_summary": "",
          "col_title": ticker_symbol,
          "metrics": ["無資料"] * 12,
          "raw": {},
      }

    company_name = (
        info.get("longName")
        or info.get("shortName")
        or ticker_symbol
    )
    short_name = info.get("shortName") or company_name
    what_it_does, future_rd, recent_news = extract_business_and_rd(info, stock)

    currency = info.get("currency", "")
    sector = info.get("sector", "N/A")
    industry = info.get("industry", "N/A")
    country = info.get("country", "N/A")
    city = info.get("city", "")
    location_str = f"{country}・{city}" if city and city != "N/A" else str(country)
    website = info.get("website", "N/A")
    employees = info.get("fullTimeEmployees")
    employees_str = f"{int(employees):,} 人" if isinstance(employees, (int, float)) and pd.notna(employees) else "N/A"

    market_cap_str = _format_large_num(info.get("marketCap"), currency=currency)
    cash_str = _format_large_num(info.get("totalCash"), currency=currency)
    total_debt_str = _format_large_num(info.get("totalDebt"), currency=currency)
    fcf_str = _format_large_num(info.get("freeCashflow"), currency=currency)
    rev_growth_str = _format_large_num(info.get("revenueGrowth"), is_pct=True)
    roe_str = _format_large_num(info.get("returnOnEquity"), is_pct=True)
    div_yield_str = _format_large_num(info.get("dividendYield"), is_pct=True)

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
        ca_str = _format_large_num(ca_val, currency=currency)
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

    profile = {
        "產業 / 細分行業": f"{sector} / {industry}",
        "總部所在地": location_str,
        "員工人數": employees_str,
        "市值規模": market_cap_str,
        "現金與約當現金": cash_str,
        "總負債": total_debt_str,
        "自由現金流": fcf_str,
        "營收成長率 (YoY)": rev_growth_str,
        "股東權益報酬率 (ROE)": roe_str,
        "股息殖利率": div_yield_str,
        "官方網站": website,
    }

    return {
        "company_name": company_name,
        "what_it_does": what_it_does,
        "future_rd": future_rd,
        "profile": profile,
        "recent_news": recent_news,
        "long_summary": info.get("longBusinessSummary", "") or "",
        "col_title": col_title,
        "metrics": [
            what_it_does,
            future_rd,
            market_cap_str,
            eps_str,
            pe_str,
            gm_str,
            pm_str,
            roe_str,
            rev_growth_str,
            ca_str,
            debt_ratio_str,
            fcf_str,
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
        "what_it_does": "無資料",
        "future_rd": "無資料",
        "profile": {},
        "recent_news": [],
        "long_summary": "",
        "col_title": ticker_symbol,
        "metrics": ["無資料"] * 12,
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
    work_df["Estimated_High"] = work_
