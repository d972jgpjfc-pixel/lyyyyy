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
    if any
