import requests
import pandas as pd
import time
from bs4 import BeautifulSoup
from datetime import datetime
import urllib3
import html # 在代码开头引入



# 关闭 SSL 证书警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ================= 核心配置 =================
START_DATE = datetime(2024, 1, 1)    # 起始日（下限）
END_DATE   = datetime(2024, 8, 31)   # 决策基准日（上限）
TARGET_COUNT = 50                    # 目标抓取条数 (20~50条)

API_URL = "https://www.chinapv.org.cn/Handler/common.ashx"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
}
# ============================================

articles = []
page = 1
reach_earlier_data = False

print(f"🚀 开始抓取 [ {START_DATE.strftime('%Y-%m-%d')} ~ {END_DATE.strftime('%Y-%m-%d')} ] 区间内的决策信息...\n")

while len(articles) < TARGET_COUNT and not reach_earlier_data:
    print(f"正在扫描第 {page} 页...")
    
    data = {
        "action": "news_list",
        "t_id": "7",
        "pageIndex": str(page),
        "pageSize": "15"  # 一页 15 条加速检索
    }
    
    try:
        response = requests.post(API_URL, headers=HEADERS, data=data, verify=False, timeout=10)
        json_data = response.json()
        page_list = json_data.get('PageList', [])
        
        if not page_list:
            print("已经翻到最后一页。")
            break
            
        for item in page_list:
            # 1. 解析时间
            raw_time = item.get('n_time', '')
            if not raw_time:
                continue
                
            date_str = raw_time.split('T')[0]
            pub_date = datetime.strptime(date_str, "%Y-%m-%d")
            
            # 2. 区间过滤
            if pub_date > END_DATE:
                # 晚于 2024-08-31，属于未来信息，跳过
                continue
            elif pub_date < START_DATE:
                # 早于 2024-01-01，说明已经翻过 2024 年了，触发刹车
                reach_earlier_data = True
                print(f"\n已扫描到 2024 年以前的数据 ({date_str})，停止翻页。")
                break
                
            # 3. 提取字段
            title = item.get('n_title', '')
            n_id = item.get('n_id', '')
            url = f"https://www.chinapv.org.cn/StaticPage/Association/content_{n_id}.html"
            
            # 清洗正文 HTML
            raw_content = item.get('n_content', '')
            unescaped_html = html.unescape(raw_content)
            clean_content = BeautifulSoup(unescaped_html, "lxml").get_text(" ", strip=True)
            
            if len(clean_content) < 30:
                continue
                
            # 4. 自动归类 (4 类核心信息)
            category = "光伏需求"
            if any(w in title for w in ["价格", "硅料", "硅片", "单晶", "多晶", "成本", "报价"]):
                category = "硅料/硅棒价格"
            elif any(w in title for w in ["政策", "补贴", "工信部", "能源局", "规划", "座谈会", "意见", "标准"]):
                category = "产业政策"
            elif any(w in title for w in ["电价", "用电", "能源", "煤炭", "绿电"]):
                category = "能源成本"

            # 5. 入库
            articles.append({
                "publish_date": date_str,
                "source": "中国光伏行业协会",
                "title": title,
                "url": url,
                "category": category,
                "content": clean_content,
                "crawl_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            print(f"  [+] 成功入库: {date_str} | [{category}] {title}")
            
            if len(articles) >= TARGET_COUNT:
                break
                
    except Exception as e:
        print(f"第 {page} 页请求异常: {e}")
        
    page += 1
    time.sleep(0.5)

# ================= 输出保存 =================
if articles:
    df = pd.DataFrame(articles)
    file_name = "external_info_2024_H1.csv"
    df.to_csv(file_name, index=False, encoding="utf-8-sig")
    print(f"\n🎉 采集完成！共获取 {len(df)} 条 2024 年 1~8 月的有效决策信息，已保存至 {file_name}")
else:
    print("\n未找到 2024年1月~8月 区间内的数据，请检查接口返回字段。")