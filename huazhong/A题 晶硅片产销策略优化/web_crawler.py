import json
import urllib.request

MODEL = "qwen3.5:4b"

PROMPT = """
你是晶硅片产销决策系统中的外部信息结构化模块。

识别：
D = 晶硅片市场需求相关信号
R = 上游原材料/单晶方棒相关价格信号
O = 无法可靠映射到D/R的信息

规则：
1. 只依据原文，不得过度推断。
2. 产量增长不能自动判断为需求增长。
3. 政策利好不能自动判断为需求增长。
4. 组件/电池出口增长可以作为需求的间接代理，但confidence不得超过2。
5. 上游硅料价格可作为单晶方棒成本方向性参考，但不得直接照搬涨跌幅。
6. 如果一句话包含多个事实，要拆开判断。
7. 如果存在有效D/R，不输出无关O。
8. 若全文没有D/R，最多输出一个O。
9. O的direction固定为0，strength固定为1。
10. evidence必须来自原文。
11. 只输出合法JSON，不要解释。

输出格式：
{
  "signals": [
    {
      "factor": "D",
      "direction": 1,
      "strength": 2,
      "confidence": 2,
      "evidence": "原文事实"
    }
  ]
}
"""

texts = [
    "光伏硅料亏本甩卖，吨价已较高点下跌近九成。",

    "2024年7月9日，工业和信息化部公开征求对《光伏制造行业规范条件（2024年本）》及《光伏制造行业规范公告管理办法（2024年本）》的意见。",

    """2024年上半年，我国光伏产业链主要环节产量均实现高比例增长。
全国光伏多晶硅、硅片、电池、组件产量同比增长均超过30%，
光伏组件出口量同比增长近20%。
硅片产量达402GW，同比增长58.6%；出口量达38.3GW。
组件产量达271GW，同比增长32.8%；组件出口量达129.2GW，同比增长19.7%。"""
]


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
                "content": "请分析下面文本：\n\n" + text
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

    with urllib.request.urlopen(req, timeout=180) as response:
        result = json.loads(response.read().decode("utf-8"))

    content = result["message"]["content"]
    return json.loads(content)


for i, text in enumerate(texts, start=1):
    print(f"\n========== 文本 {i} ==========")

    try:
        result = ask_qwen(text)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print("运行失败：", e)