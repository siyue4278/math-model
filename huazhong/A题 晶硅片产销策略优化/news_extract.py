import re
import time
import requests
import pandas as pd
import trafilatura

from bs4 import BeautifulSoup


# =========================================================
# 1. 候选新闻清单
# =========================================================

ARTICLES = [
    {
        "date": "2024-01-25",
        "source": "硅业分会转载",
        "title": "硅料价格小幅上涨",
        "url": "https://m.energytrend.cn/news/20240125-133084.html"
    },
    {
        "date": "2024-02-21",
        "source": "财联社/硅业分会",
        "title": "多晶硅节后价格持稳",
        "url": "https://www.cls.cn/detail/1600265"
    },
    {
        "date": "2024-03-07",
        "source": "硅业分会转载",
        "title": "新增产能进入市场、硅料价格波动",
        "url": "https://m.solarzoom.com/index.php/article/183159"
    },
    {
        "date": "2024-03-21",
        "source": "硅业分会转载",
        "title": "上下游博弈、成交清淡",
        "url": "https://m.energytrend.cn/news/20240321-134472.html"
    },
    {
        "date": "2024-03-28",
        "source": "中证网/硅业分会",
        "title": "硅片供应过剩、价格下跌",
        "url": "https://www.cs.com.cn/cj2020/202403/t20240328_6397913.html"
    },
    {
        "date": "2024-04-12",
        "source": "硅业分会转载",
        "title": "多晶硅价格全线下调",
        "url": "https://m.energytrend.cn/news/20240412-135099.html"
    },
    {
        "date": "2024-05-31",
        "source": "硅业分会转载",
        "title": "N型硅料再降、企业检修",
        "url": "https://m.energytrend.cn/news/20240531-136335.html"
    },
    {
        "date": "2024-06-05",
        "source": "硅业分会转载",
        "title": "多晶硅成交低迷、主动减产",
        "url": "https://finance.sina.cn/hkstock/gsxw/2024-06-05/detail-inaxsvtz6837986.d.html"
    },
    {
        "date": "2024-06-20",
        "source": "硅业分会转载",
        "title": "多晶硅价格逼近底部",
        "url": "https://www.ne21.com/news/show-196366.html"
    },
    {
        "date": "2024-06-27",
        "source": "财联社/硅业分会",
        "title": "多晶硅价格再度下调",
        "url": "https://www.cls.cn/detail/1716039"
    },
    {
        "date": "2024-07-11",
        "source": "硅业分会转载",
        "title": "多晶硅价格无上涨动力",
        "url": "https://m.energytrend.cn/news/20240711-137276.html"
    },
    {
        "date": "2024-08-01",
        "source": "硅业分会转载",
        "title": "市场活跃度提升、小单探涨",
        "url": "https://m.energytrend.cn/news/20240801-137759.html"
    },
    {
        "date": "2024-04-29",
        "source": "国家能源局",
        "title": "2024年一季度可再生能源并网运行情况",
        "url": "https://www.nea.gov.cn/2024-04/29/c_1212357856.htm"
    },
    {
        "date": "2024-05-06",
        "source": "国家能源局",
        "title": "2024年一季度光伏发电建设情况",
        "url": "https://www.nea.gov.cn/2024-05/06/c_1310773741.htm"
    },
    {
        "date": "2024-07-09",
        "source": "工业和信息化部",
        "title": "光伏制造行业规范条件（2024年本）征求意见",
        "url": "https://www.miit.gov.cn/jgsj/dzs/gzdt/art/2024/art_16a86cc6229d4cf8b52c54dd01df7a74.html"
    },
    {
        "date": "2024-07-31",
        "source": "国家能源局",
        "title": "2024年上半年可再生能源并网运行情况",
        "url": "https://www.nea.gov.cn/2024-07/31/c_1310783334.htm"
    },
    {
        "date": "2024-08-08",
        "source": "工业和信息化部",
        "title": "2024年上半年全国光伏制造行业运行情况",
        "url": "https://www.miit.gov.cn/jgsj/dzs/gzdt/art/2024/art_3f5fc50b3bc44aba9311c79455bda8a8.html"
    }
]


# =========================================================
# 2. 与第四问相关的关键词
# =========================================================

KEYWORDS = [
    "硅片",
    "多晶硅",
    "硅料",
    "单晶方棒",
    "方棒",
    "原材料",
    "价格",
    "报价",
    "成交价",
    "成交均价",
    "需求",
    "采购",
    "订单",
    "库存",
    "装机",
    "出口",
    "开工率",
    "排产",
    "减产",
    "产能",
    "组件",
    "电池"
]


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0 Safari/537.36"
    )
}


# =========================================================
# 3. 下载网页
# =========================================================

def fetch_html(url):
    response = requests.get(
        url,
        headers=HEADERS,
        timeout=20
    )

    response.raise_for_status()

    # 尽量避免中文乱码
    if response.encoding is None or response.encoding.lower() == "iso-8859-1":
        response.encoding = response.apparent_encoding

    return response.text


# =========================================================
# 4. 正文提取
#    第一优先级：trafilatura
#    第二优先级：BeautifulSoup
# =========================================================

def extract_text(html):
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        include_links=False,
        favor_precision=True
    )

    if text and len(text.strip()) >= 100:
        return text.strip(), "trafilatura"

    # fallback
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup([
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "aside"
    ]):
        tag.decompose()

    paragraphs = []

    for p in soup.find_all("p"):
        t = p.get_text(" ", strip=True)

        if len(t) >= 15:
            paragraphs.append(t)

    text = "\n".join(paragraphs)

    if len(text.strip()) >= 100:
        return text.strip(), "beautifulsoup"

    return "", "failed"


# =========================================================
# 5. 基础文本清洗
# =========================================================

def clean_paragraph(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================================================
# 6. 根据关键词选取相关段落
#
# 找到命中关键词的段落后，
# 同时保留前后各1段，保证Qwen拥有上下文。
# =========================================================

def select_relevant_text(text, context=1, max_chars=3500):

    paragraphs = [
        clean_paragraph(p)
        for p in text.split("\n")
        if len(clean_paragraph(p)) >= 10
    ]

    hit_indexes = []

    for i, paragraph in enumerate(paragraphs):

        if any(keyword in paragraph for keyword in KEYWORDS):
            hit_indexes.append(i)

    if not hit_indexes:
        return ""

    selected_indexes = set()

    for i in hit_indexes:

        start = max(0, i - context)
        end = min(len(paragraphs), i + context + 1)

        for j in range(start, end):
            selected_indexes.add(j)

    selected = [
        paragraphs[i]
        for i in sorted(selected_indexes)
    ]

    result = "\n".join(selected)

    # 防止一篇文章过长
    return result[:max_chars]


# =========================================================
# 7. 主程序
# =========================================================

def main():

    results = []

    total = len(ARTICLES)

    for index, article in enumerate(ARTICLES, start=1):

        print(
            f"\n[{index}/{total}] "
            f"{article['date']} "
            f"{article['title']}"
        )

        try:

            html = fetch_html(article["url"])

            full_text, extractor = extract_text(html)

            if not full_text:

                status = "extract_failed"
                relevant_text = ""

                print("  × 正文提取失败")

            else:

                relevant_text = select_relevant_text(full_text)

                if relevant_text:

                    status = "ok"

                    print(
                        f"  √ 提取成功 | "
                        f"{extractor} | "
                        f"全文 {len(full_text)} 字 | "
                        f"筛选 {len(relevant_text)} 字"
                    )

                else:

                    status = "no_keyword"

                    print(
                        f"  △ 正文提取成功，但未找到相关关键词"
                    )

            results.append({
                "date": article["date"],
                "source": article["source"],
                "title": article["title"],
                "url": article["url"],
                "status": status,
                "extractor": extractor,
                "full_text_length": len(full_text),
                "relevant_text": relevant_text
            })

        except Exception as e:

            print("  × 请求失败：", e)

            results.append({
                "date": article["date"],
                "source": article["source"],
                "title": article["title"],
                "url": article["url"],
                "status": "request_failed",
                "extractor": "",
                "full_text_length": 0,
                "relevant_text": "",
                "error": str(e)
            })

        # 不要短时间疯狂访问网站
        time.sleep(1.5)

    # =====================================================
    # 保存全部结果
    # =====================================================

    df = pd.DataFrame(results)

    df.to_csv(
        "news_extract.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # =====================================================
    # 单独保存失败网页
    # =====================================================

    failed = df[df["status"] != "ok"]

    failed.to_csv(
        "failed_urls.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n===================================")
    print("正文提取结束")
    print("===================================")

    print(f"总文章数：{len(df)}")
    print(f"成功：{sum(df['status'] == 'ok')}")
    print(f"失败/需检查：{sum(df['status'] != 'ok')}")

    print("\n生成文件：")
    print("news_extract.csv")
    print("failed_urls.csv")


if __name__ == "__main__":
    main()