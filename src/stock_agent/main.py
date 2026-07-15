import os
import sys
import json
import schedule
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from openai import OpenAI
from sklearn.linear_model import LinearRegression

MEMORY_FILE = "agent_memory.json"

def load_stock_memory(ticker: str) -> dict:
    """从本地文件读取指定股票的记忆历史"""
    if not os.path.exists(MEMORY_FILE):
        return {}
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            full_memory = json.load(f)
            return full_memory.get(ticker, {})
    except Exception as e:
        print(f" 读取记忆库失败: {e}")
        return {}

def save_stock_memory(ticker: str, date_str: str, actual_close: float, trend: str, predicted_prices: list):
    """保存今天的行情和预测到记忆库中"""
    full_memory = {}
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                full_memory = json.load(f)
        except Exception:
            full_memory = {}
            
    if ticker not in full_memory:
        full_memory[ticker] = {}
        
    # 以日期为 Key 记录
    full_memory[ticker][date_str] = {
        "actual_close": actual_close,
        "predicted_trend": trend,
        "predicted_prices": predicted_prices
    }
    
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(full_memory, f, ensure_ascii=False, indent=4)
        print(f" 💾 今日量化数据与预测已存入智能体长期记忆库。")
    except Exception as e:
        print(f" 写入记忆库失败: {e}")

def reflect_on_previous_prediction(ticker: str, today_close: float) -> str:
    """
    【反思引擎】读取上一次的预测，并与今天真实收盘价对比，计算误差
    """
    memory = load_stock_memory(ticker)
    if not memory:
        return "【复盘提示】这是智能体第一次对该股进行观测，暂无过往记忆可供复盘。"
        
    # 获取最近的一个历史预测记录（排除今天）
    sorted_dates = sorted(memory.keys())
    if not sorted_dates:
        return "【复盘提示】暂无历史复盘数据。"
        
    last_record_date = sorted_dates[-1]
    last_record = memory[last_record_date]
    
    pred_prices = last_record.get("predicted_prices", [])
    pred_trend = last_record.get("predicted_trend", "未知")
    
    if not pred_prices:
        return "【复盘提示】上次未录入预测价格，跳过复盘。"
        
    # 拿出上次预测列表中对“未来第1天”（也就是今天）的预测价格
    yesterday_pred_for_today = pred_prices[0]
    
    # 计算绝对误差和百分比
    error_val = today_close - yesterday_pred_for_today
    error_percent = (abs(error_val) / today_close) * 100
    
    reflection_text = f"""
    【智能体自我复盘历史记录】
    前一个观测交易日({last_record_date})的预测回顾：
    - 当时研判的短期趋势为: {pred_trend}
    - 当时模型预测今日(T+1)的收盘价为: {yesterday_pred_for_today:.2f}
    - 今日实际真实收盘价为: {today_close:.2f}
    - 预测绝对误差: {error_val:+.2f} 美元，误差率: {error_percent:.2f}%
    """
    return reflection_text

def get_stock_data(ticker_symbol: str) -> tuple[pd.DataFrame, dict]:
    """获取指定股票（支持美股与A股）的最新历史行情与基础财务指标"""
    print(f"【1/6】正在从 yfinance 获取 {ticker_symbol} 的最新原始数据...")
    try:
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        stock = yf.Ticker(ticker_symbol, session=session)
        df = stock.history(period="1y")

        if df.empty:
            raise ValueError(f"未能获取到 {ticker_symbol} 的历史行情数据。")

        # 提取基本面指标（针对 A 股做了安全的增强获取）
        info = stock.info
        
        # 优先获取中文名或长名称
        company_name = info.get("shortName") or info.get("longName") or info.get("symbol", "N/A")
        
        key_metrics = {
            "公司名称": company_name,
            "行业": info.get("industry", "N/A"),
            "市值": info.get("marketCap", "N/A"),
            "前瞻市盈率(Forward PE)": info.get("forwardPE", "N/A"),
            "市净率(Price to Book)": info.get("priceToBook", "N/A"),
            "52周最高价": info.get("fiftyTwoWeekHigh", "N/A"),
            "52周最低价": info.get("fiftyTwoWeekLow", "N/A"),
            "分红率(Dividend Yield)": info.get("dividendYield", "N/A")
        }

        latest_market_date = df.index[-1].strftime('%Y-%m-%d')
        print(f" 成功获取数据！最新交易日为: {latest_market_date}")
        
        return df, key_metrics

    except Exception as e:
        print(f"获取数据失败: {e}")
        return pd.DataFrame(), {}

def get_stock_news(ticker_symbol: str, model_name: str, llm_config: dict) -> str:
    """【智能修改】获取指定股票今天和昨天的财经新闻，并多次调用大模型逐条评估媒体可信度与内容相关性"""
    print(f"【4/6】正在获取 {ticker_symbol} 近两日财经新闻并利用大模型智能筛选可信度...")
    
    api_key = llm_config.get("llm_api_key")
    base_url = llm_config.get("llm_base_url", "https://api.deepseek.com")
    
    if not api_key:
        return "未配置 LLM API Key，跳过新闻筛选。"

    try:
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        stock = yf.Ticker(ticker_symbol, session=session)
        news_list = stock.news
        
        if not news_list:
            return "暂未获取到近期公开的重大财经新闻。"
            
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        
        valid_news_count = 0
        trusted_news_list = []
        
        # 逐条审查新闻时效性与来源可信度
        for item in news_list:
            # 限制单次任务最多初筛5条最新新闻，防止调用大模型次数过多导致程序过度缓慢
            if valid_news_count >= 5:
                break
                
            pub_time_epoch = item.get("providerPublishTime")
            if not pub_time_epoch:
                continue
                
            pub_date = datetime.fromtimestamp(pub_time_epoch).date()
            
            # 严格时效限制：只需要考虑今天和昨天的新闻
            if pub_date < yesterday:
                continue
                
            valid_news_count += 1
            title = item.get("title", "无标题")
            publisher = item.get("publisher", "未知媒体")
            
            # 针对单条新闻，调用大模型充当合规与媒体可信度裁判
            system_prompt = """
            你是一个专业的财经新闻审查官。请严格评估所提供新闻的【媒体来源可信度】以及【内容对目标股票及其产业链的实质性影响】。
            
            评估准则：
            1. 媒体可信度：主流财经媒体（如Bloomberg, Reuters, WSJ, CNBC, 财新, 证券时报, 每日经济新闻等）或权威机构判定为高。自媒体吹水、八卦小报、软文推广判定为低。
            2. 内容相关性（必须具有核心参考价值，满足以下其一即可）：
               - 【企业本体】：直接涉及该公司的基本面、业绩财报、核心高管变动、重大商业合作或公关危机。
               - 【所在行业】：涉及该企业所属行业的重大政策出台、颠覆性技术突破、整体市场需求变化或系统性风险。
               - 【关联产业链】：涉及该企业上下游（如关键原材料价格暴涨/暴跌、供应链中断）、核心竞品的大动作，或宏观经济事件对该行业存在明确的“传导效应”。
            
            输出规范：
            - 如果媒体来源不可信，或者内容对目标企业、所在行业及产业链毫无实质性波及（判定为噪音），请直接精准回复四个字：【放弃忽略】
            - 如果媒体权威可信且内容具有直接或间接的投资参考价值，请直接输出格式：[可信来源 - 媒体名称] 用一句大白话总结该新闻的核心内容及其对该公司的潜在影响。严禁使用任何复杂的专业术语，必须简单好懂。
            """
            
            user_content = f"目标股票代码: {ticker_symbol}\n新闻发布媒体: {publisher}\n新闻标题内容: {title}"
            
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.2,  # 低随机性，确保严格遵照输出规范
                )
                res_text = response.choices[0].message.content.strip()
                if "放弃忽略" not in res_text:
                    trusted_news_list.append(res_text)
            except Exception as eval_e:
                print(f" 评估单条新闻来源 [{publisher}] 失败: {eval_e}")
                # 降级容错方案：接口偶尔超时则退化保留原数据
                trusted_news_list.append(f"[{publisher}] {title} (未完成智能可信度筛查)")
        
        if not trusted_news_list:
            return "近两天内有零星新闻，但经大模型可信度与相关性智能初筛，均判定为低价值自媒体噪音，已自动忽略。"
            
        return "\n".join([f"{idx}. {news}" for idx, news in enumerate(trusted_news_list, 1)])
        
    except Exception as e:
        print(f" 获取或处理新闻失败: {e}")
        return "获取最新新闻流失败，跳过新闻参考。"

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """基于历史行情数据计算常用的技术指标 (MA, RSI, MACD)"""
    print("【2/6】正在计算量化技术指标 (MA5, MA20, RSI, MACD)...")
    df = df.copy()

    # 1. 计算移动平均线
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA20"] = df["Close"].rolling(window=20).mean()

    # 2. 计算相对强弱指标 (RSI - 14天)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)  # 防止除以0
    df["RSI"] = 100 - (100 / (1 + rs))

    # 3. 计算 MACD (12, 26, 9)
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Hist"] = df["MACD"] - df["Signal"]

    return df

def predict_trend(df: pd.DataFrame) -> dict:
    """使用线性回归模型预测未来 3 天的股价趋势与拟合收盘价"""
    print("【3/6】运行机器学习线性回归算法，研判短期价格斜率与预测...")
    if len(df) < 15:
        return {"trend": "数据不足", "slope": 0, "predicted_prices": []}

    # 取最近 15 个交易日的收盘价作为特征构建
    recent_prices = df["Close"].tail(15).values.reshape(-1, 1)
    X = np.arange(len(recent_prices)).reshape(-1, 1)

    # 训练线性模型
    model = LinearRegression()
    model.fit(X, recent_prices)

    # 斜率 (Slope) 代表价格趋势的方向和强度
    slope = model.coef_[0][0]

    # 预测未来 3 个交易日的价格趋势
    future_X = np.array([[15], [16], [17]])
    future_preds = model.predict(future_X).flatten().tolist()

    if slope > 0.5:
        trend_desc = "强劲上涨趋势"
    elif slope > 0.1:
        trend_desc = "温和上涨趋势"
    elif slope < -0.5:
        trend_desc = "强劲下跌趋势"
    elif slope < -0.1:
        trend_desc = "温和下跌趋势"
    else:
        trend_desc = "横盘震荡"

    return {
        "trend": trend_desc,
        "slope": float(slope),
        "last_close": float(recent_prices[-1][0]),
        "predicted_prices": future_preds,
    }

def generate_report(ticker: str, metrics: dict, latest_data: dict, prediction: dict, reflection_context: str, model_name: str, news_context: str, llm_config: dict) -> str:
    """【修复Bug + 风格通俗化】调用大模型 API，结合量化指标与筛选后的新闻，生成通俗的大白话分析报告"""
    print("【5/6】唤醒大模型大脑，整合数据与筛选后的新闻撰写大白话投资分析报告...")

    api_key = llm_config.get("llm_api_key")
    base_url = llm_config.get("llm_base_url", "https://api.deepseek.com")

    if not api_key:
        return "【系统错误提示】未检测到有效的大模型 API Key。"

    client = OpenAI(api_key=api_key, base_url=base_url)

    prompt_context = f"""
    {reflection_context}
    
    【数据基准时间】
    当前报告日期: {latest_data.get('Date', '未知日期')}
    
    【股票基本信息】
    代码: {ticker} | 公司: {metrics.get('公司名称', 'N/A')}
    行业: {metrics.get('行业', 'N/A')} | 市值: {metrics.get('市值', 'N/A')}
    
    【盘面数据与技术指标】
    最新收盘价: {latest_data['Close']:.2f}
    5日均线(MA5): {latest_data['MA5']:.2f} | 20日均线(MA20): {latest_data['MA20']:.2f}
    买卖气势指标RSI: {latest_data['RSI']:.2f} | 趋势动能MACD值: {latest_data['MACD']:.2f}
    
    【机器学习短期预测】
    趋势方向与强度(Slope): {prediction['slope']:.4f}
    计算机研判短期趋势: {prediction['trend']}
    未来3个交易日电脑预测收盘价: {[round(p, 2) for p in prediction['predicted_prices']]}
    
    【经大模型初筛的近两日高可信财经新闻摘要】
    {news_context}
    """

    system_prompt = """
    你是一位接地气、说话直白透彻的华尔街投资顾问。
    请根据用户提供的精准数据、历史复盘以及筛选后的新闻，为普通股民撰写一份中文股市分析日报。
    
    核心军规（必须严格执行）：
    1. 【绝对不要用语太专业】：禁止使用晦涩难懂的量化模型定义或堆砌金融学术黑话。你的读者是普通散户，报告要非常简单。
    2. 【用大白话解释指标】：把专业技术指标转化为生活化语言。例如：MA5/MA20均线多头解释为“股价正被短期和中期持仓成本稳稳托着走，势头向上”；RSI解释为“大家有没有买过头/市场热度高不高”；MACD直接说“上涨的油门踩得深不深”。
    3. 报告必须精简，且严格包含以下模块：
       - 1. 【昨日预测复盘】：老老实实对比昨天的预测和今天的结果，说清楚算得准不准，错在哪了，别绕弯子。
       - 2. 【今天盘面大白话综述】：今天涨跌的势头怎么样，多方和空方谁占上风。
       - 3. 【新闻与电脑预测解读】：结合最新搜集到的这几条可信新闻（看看有啥实质性消息在传），再结合电脑算出来的未来3天预测价，给出一个最直观的通俗分析。
       - 4. 【明天咋操作与防守位】：直接告诉读者应该谨慎、持股还是看戏，并给出一个明确的“跌破就必须离场”的防守价格。
    
    报告开头必须标注正确的【报告日期】（严格使用用户提供的数据基准时间）。使用简洁好看的 Markdown 排版。
    """

    try:
        response = client.chat.completions.create(
            model=model_name, 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_context},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"调用大模型失败，错误详情: {e}"

def send_notification(stock_code: str, report_content: str, tg_config: dict) -> bool:
    """
    将分析报告推送到 Telegram 机器人
    """
    if not tg_config:
        print("【提示】未配置 Telegram 信息，跳过自动推送。")
        return False

    bot_token = tg_config.get("bot_token")
    chat_id = tg_config.get("chat_id")
    proxy_url = tg_config.get("proxy")
    
    if not bot_token or not chat_id:
        print("【提示】未配置 Telegram Token 或 Chat ID，跳过自动推送。")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    full_message = f"*📊 {stock_code} 智能量化分析日报*\n\n{report_content}"

    payload = {
        "chat_id": chat_id,
        "text": full_message,
        "parse_mode": "Markdown" 
    }

    try:
        proxies = None
        if proxy_url:
            proxies = {"http": proxy_url, "https": proxy_url}
        response = requests.post(url, json=payload, proxies=proxies, timeout=10)
        
        if response.status_code == 200:
            print(f" 成功推送到 Telegram！")
            return True
        else:
            print(f" 推送失败，状态码: {response.status_code}, 响应: {response.text}")
            return False
    except Exception as e:
        print(f" 推送发生异常（请检查网络与代理设置）: {e}")
        return False
    
def send_qq_notification(stock_code: str, report_content: str, qq_config: dict) -> bool:
    """
    将分析报告推送到 QQ 机器人（兼容 OneBot 11 协议如 NapCatQQ / LLOneBot）
    """
    if not qq_config:
        print("【提示】未检测到 qq_config 配置，跳过 QQ 推送。")
        return False

    api_url = qq_config.get("api_url", "http://127.0.0.1:3000").rstrip('/')
    target_group = qq_config.get("group_id", "")
    target_user = qq_config.get("user_id", "")
    token = qq_config.get("access_token", "")
    
    if not target_group and not target_user:
        print("【提示】未配置 QQ 群号或好友 QQ 号，跳过 QQ 推送。")
        return False
        
    # QQ 不支持标准 Markdown 渲染，去除一些影响阅读的星号，保持大白话排版
    clean_report = report_content.replace("**", "").replace("###", "■").replace("##", "▼")
    full_message = f"📊 {stock_code} 智能量化分析日报\n\n{clean_report}"
    
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        
    success = True
    
    # 1. 发送到指定群聊
    if target_group:
        try:
            url = f"{api_url}/send_group_msg"
            payload = {"group_id": int(target_group), "message": full_message}
            res = requests.post(url, json=payload, headers=headers, timeout=10)
            if res.status_code == 200 and res.json().get("status") == "ok":
                print(f" ✅ 成功推送到 QQ 群 [{target_group}]")
            else:
                print(f" ❌ 推送 QQ 群失败: {res.text}")
                success = False
        except Exception as e:
            print(f" ❌ 推送 QQ 群发生异常: {e}")
            success = False
            
    # 2. 发送到指定个人/好友
    if target_user:
        try:
            url = f"{api_url}/send_private_msg"
            payload = {"user_id": int(target_user), "message": full_message}
            res = requests.post(url, json=payload, headers=headers, timeout=10)
            if res.status_code == 200 and res.json().get("status") == "ok":
                print(f" ✅ 成功推送到 QQ 好友 [{target_user}]")
            else:
                print(f" ❌ 推送 QQ 好友失败: {res.text}")
                success = False
        except Exception as e:
            print(f" ❌ 推送 QQ 好友发生异常: {e}")
            success = False
            
    return success

def run_stock_agent_job(config: dict):
    """智能体核心任务流水线（动态读取外部配置 + 双日可信新闻初筛 + 通俗分析）"""
    rules = config.get("rules", {})
    
    if rules.get("skip_weekends", True):
        current_day = datetime.now().weekday()
        if current_day in [5, 6]:
            print(f"📅 当前为周末，美股休市，根据 JSON 规则跳过执行。")
            return

    watchlist = config.get("watchlist", [])
    model_name = rules.get("model_name", "deepseek-v4-pro")
    tg_config = config.get("tg_config", {})
    qq_config = config.get("qq_config", {})
    
    print(f"\n⏰ 触发定时任务：开始执行每日量化分析 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
    
    for target_stock in watchlist:
        print(f"\n================ 正在处理: {target_stock} ================")
        try:
            # 1. 抓取行情数据
            df, metrics = get_stock_data(target_stock)
            if df.empty:
                continue
                
            # 2. 计算指标
            df_with_indicators = calculate_indicators(df)
            
            # 3. 机器学习预测
            prediction_results = predict_trend(df_with_indicators)
            
            # 4. 收集财经新闻（内部已包含多大模型调用及今明两日/媒体可信度筛选逻辑）
            news_context = get_stock_news(target_stock, model_name, rules)
            
            latest_row = df_with_indicators.iloc[-1]
            market_date_str = latest_row.name.strftime('%Y-%m-%d') if hasattr(latest_row.name, 'strftime') else str(latest_row.name)
            
            latest_data_snapshot = {
                "Date": market_date_str,
                "Close": float(latest_row['Close']),
                "MA5": float(latest_row['MA5']),
                "MA20": float(latest_row['MA20']),
                "RSI": float(latest_row['RSI']),
                "MACD": float(latest_row['MACD'])
            }
            
            # 5. 读取历史记忆进行反思
            reflection_context = reflect_on_previous_prediction(target_stock, latest_data_snapshot["Close"])
            
            # 6. 大模型综合撰写大白话报告
            report = generate_report(target_stock, metrics, latest_data_snapshot, prediction_results, reflection_context, model_name, news_context, rules)
            
            # 保存本地报告
            if rules.get("save_local_report", True):
                safe_filename = target_stock.replace('.', '_')
                report_filename = f"{safe_filename}_daily_report.md"
                with open(report_filename, "w", encoding="utf-8") as f:
                    f.write(report)
                print(f"💾 本地文件保存成功: {report_filename}")
                
            # 推送 Telegram
            if tg_config.get("enabled", False):
                print(f"【6/6】正在将 {target_stock} 的日报推送到 Telegram...")
                send_notification(target_stock, report, tg_config)
            
            # 【新增：推送到 QQ 机器人】
            if qq_config.get("enabled", False):
                print(f" 正在将 {target_stock} 的日报推送到 QQ 机器人...")
                send_qq_notification(target_stock, report, qq_config)

            # 保存记忆
            save_stock_memory(
                ticker=target_stock, 
                date_str=market_date_str, 
                actual_close=latest_data_snapshot["Close"], 
                trend=prediction_results["trend"], 
                predicted_prices=prediction_results["predicted_prices"]
            )
            
        except Exception as e:
            print(f"❌ 错误: 处理 {target_stock} 时发生未知异常: {e}")
            
    print("\n🎉 本轮任务全部结束。")

def load_config(config_path="config.json") -> dict:
    """从外部 JSON 文件读取智能体配置，若文件不存在或解析失败则返回硬编码默认配置"""
    default_config = {
        "watchlist": ["AAPL", "NVDA"],
        "schedule_time": "06:00",
        "rules": {
            "skip_weekends": True,
            "model_name": "deepseek-v4-pro",
            "save_local_report": True
        }
    }
    if not os.path.exists(config_path):
        print(f"⚠️ 未检测到配置文件 {config_path}，使用系统默认内置配置。")
        return default_config
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 解析配置文件 {config_path} 失败: {e}，回退至内置配置。")
        return default_config

def main():
    config = load_config()
    
    # 模式一：命令行强制单次手动触发
    if len(sys.argv) > 1 and sys.argv[1] == "--now":
        print("🚀 检测到命令行参数 [--now]，正在紧急手动触发单次量化分析流水线...\n")
        run_stock_agent_job(config)
        print("\n✨ 单次手动触发流水线执行完毕，程序退出。")
        sys.exit(0)

    # 模式二：标准后台守护进程模式
    print("🤖 智能体后台守护进程已启动...")
    target_time = config.get("schedule_time", "06:00")
    
    schedule.every().day.at(target_time).do(run_stock_agent_job, config=config)
    
    print(f"📅 任务已排期：将在每天北京时间 {target_time} 自动读取最新数据并推送 Telegram。")
    print("💡 提示：若想直接手动触发单次任务，请在终端运行: stock-agent --now")
    print("----------------------------------------------------------------------------------")
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()
