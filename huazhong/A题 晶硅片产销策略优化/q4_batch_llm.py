import json
import time
import urllib.request
import pandas as pd


# =========================================================
# 1. 基本配置
# =========================================================

MODEL = "qwen3.5:4b"

INPUT_FILE = "news_extract.csv"
OUTPUT_FILE = "qwen_signals.csv"
RAW_FILE = "qwen_raw_results.csv"


# =========================================================
# 2. 固定 Prompt
# =========================================================

PROMPT = """
你是晶硅片产销决策系统中的外部信息结构化模块。

识别：
D = 晶硅片市场需求相关信号
R = 上游原材料/单晶方棒相关价格信号
O = 无法可靠映射到D/R的信息

每个信号包含：
- factor：D、R 或 O
- direction：-1、0、1
- strength：1、2、3
- confidence：1、2、3
- evidence：原文中的简短事实依据

【D 判断规则】
1. 明确的需求、采购需求、订单增加/减少等，可以判为D。
2. 下游采购量、订单量、终端装机量等可作为需求指标。
3. 电池、组件出口增长等可作为间接需求代理，但confidence不得超过2。
4. 产量增长、扩产、产能增长不能单独判定为需求增长。
5. 政策利好不能自动判定为需求增长。
6. 单纯库存变化而没有需求说明，不能直接判定为D。

【R 判断规则】
1. 明确涉及多晶硅、硅料、单晶方棒等上游原料价格、
   成交均价、报价、涨价、降价等信息时，可以判为R。
2. 上游硅料价格可以作为单晶方棒成本的方向性参考，
   但不能将硅料涨跌幅直接等同于单晶方棒涨跌幅。

【事实拆分】
如果一句话包含多个独立事实，必须分别判断。
例如：
“组件产量增长32.8%，组件出口增长19.7%”
其中产量增长不能单独判D，
出口增长可以作为D的间接代理。

【O 判断规则】
1. 无法可靠映射到D/R时才输出O。
2. 如果已经发现有效D或R，不需要把其他无关信息逐条输出O。
3. 如果全文没有D/R，最多输出一个O。
4. O的direction固定为0，strength固定为1。

严格遵守：
1. 只依据原文证据，不得过度推断。
2. direction表示因素本身方向，不表示对利润的影响。
3. evidence必须能在原文中直接找到。
4. 不预测具体销量比例、价格、利润或生产计划。
5. 只输出合法JSON。
6. 不输出解释、前言、markdown代码块或任何JSON之外的文字。

输出格式：
{
  "signals": [
    {
      "factor": "D",
      "direction": 1,
      "strength": 2,
      "confidence": 2,
      "evidence": "原文中的事实"
    }
  ]
}
"""


# =========================================================
# 3. 调用本地 Ollama
# =========================================================

def ask_qwen(text):

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": PROMPT
            },
            {
                "role": "user",
                "content": "请分析下面文本：\n\n" + str(text)
            }
        ],
        "stream": False,
        "think": False,
        "format": "json"
    }

    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(req, timeout=240) as response:
        result = json.loads(
            response.read().decode("utf-8")
        )

    content = result["message"]["content"]

    return content


# =========================================================
# 4. JSON解析
# =========================================================

def parse_result(raw):

    try:
        obj = json.loads(raw)

        signals = obj.get("signals", [])

        if not isinstance(signals, list):
            return []

        return signals

    except Exception:
        return []


# =========================================================
# 5. 确定性规则清洗
# =========================================================

def clean_signals(signals):

    cleaned = []

    for s in signals:

        if not isinstance(s, dict):
            continue

        factor = str(s.get("factor", "")).upper()

        if factor not in ["D", "R", "O"]:
            continue

        try:
            direction = int(s.get("direction", 0))
            strength = int(s.get("strength", 1))
            confidence = int(s.get("confidence", 1))
        except Exception:
            continue

        # 限制合法范围
        direction = max(-1, min(1, direction))
        strength = max(1, min(3, strength))
        confidence = max(1, min(3, confidence))

        evidence = str(s.get("evidence", "")).strip()

        # O强制规范
        if factor == "O":
            direction = 0
            strength = 1

        cleaned.append({
            "factor": factor,
            "direction": direction,
            "strength": strength,
            "confidence": confidence,
            "evidence": evidence
        })

    # 如果已经存在D/R，则删除所有O
    has_valid_signal = any(
        s["factor"] in ["D", "R"]
        for s in cleaned
    )

    if has_valid_signal:
        cleaned = [
            s for s in cleaned
            if s["factor"] in ["D", "R"]
        ]

    else:
        # 没有D/R时最多保留一个O
        os = [
            s for s in cleaned
            if s["factor"] == "O"
        ]

        cleaned = os[:1]

    return cleaned


# =========================================================
# 6. 主程序
# =========================================================

def main():

    df = pd.read_csv(INPUT_FILE)

    # 只处理正文提取成功的文章
    df = df[df["status"] == "ok"].copy()

    print(f"准备分析文章数：{len(df)}")

    signal_rows = []
    raw_rows = []

    for count, (_, row) in enumerate(df.iterrows(), start=1):

        print()
        print("=" * 60)
        print(
            f"[{count}/{len(df)}] "
            f"{row['date']} | {row['title']}"
        )

        text = row["relevant_text"]

        try:

            raw = ask_qwen(text)

            print("Qwen返回成功")

            signals = parse_result(raw)

            cleaned = clean_signals(signals)

            print(
                "识别结果：",
                [
                    (
                        s["factor"],
                        s["direction"],
                        s["strength"],
                        s["confidence"]
                    )
                    for s in cleaned
                ]
            )

            # 保存原始输出，方便论文追溯
            raw_rows.append({
                "date": row["date"],
                "source": row["source"],
                "title": row["title"],
                "url": row["url"],
                "raw_json": raw,
                "status": "success"
            })

            # 长表：一个signal一行
            if cleaned:

                for s in cleaned:

                    signal_rows.append({
                        "date": row["date"],
                        "source": row["source"],
                        "title": row["title"],
                        "url": row["url"],
                        "factor": s["factor"],
                        "direction": s["direction"],
                        "strength": s["strength"],
                        "confidence": s["confidence"],
                        "evidence": s["evidence"]
                    })

            else:

                print("警告：没有得到有效signal")

        except Exception as e:

            print("分析失败：", e)

            raw_rows.append({
                "date": row["date"],
                "source": row["source"],
                "title": row["title"],
                "url": row["url"],
                "raw_json": "",
                "status": "failed",
                "error": str(e)
            })

        # 本地模型不用太快连续压请求
        time.sleep(0.5)

    # =====================================================
    # 保存结果
    # =====================================================

    signals_df = pd.DataFrame(signal_rows)
    raw_df = pd.DataFrame(raw_rows)

    signals_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    raw_df.to_csv(
        RAW_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 60)
    print("批量分析完成")
    print("=" * 60)

    print(f"文章数：{len(df)}")
    print(f"提取signal数：{len(signals_df)}")

    if not signals_df.empty:

        print()
        print("因子数量：")
        print(signals_df["factor"].value_counts())

        print()
        print("方向分布：")
        print(
            signals_df.groupby(
                ["factor", "direction"]
            ).size()
        )

    print()
    print("生成文件：")
    print(OUTPUT_FILE)
    print(RAW_FILE)


if __name__ == "__main__":
    main()