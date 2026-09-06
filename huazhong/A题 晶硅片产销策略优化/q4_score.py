import pandas as pd


INPUT_FILE = "qwen_signals.csv"

df = pd.read_csv(INPUT_FILE)

# 只保留真正进入数学模型的 D / R
df = df[df["factor"].isin(["D", "R"])].copy()

# ---------------------------------------------------------
# 1. 单条信号的分子、分母
# ---------------------------------------------------------

df["num"] = (
    df["direction"]
    * df["strength"]
    * df["confidence"]
)

df["den"] = (
    3 * df["confidence"]
)


# ---------------------------------------------------------
# 2. 先聚合到“文章-因子”层面
#    防止一篇文章输出很多signals而权重过大
# ---------------------------------------------------------

article_scores = (
    df.groupby(
        ["date", "source", "title", "factor"],
        as_index=False
    )
    .agg(
        numerator=("num", "sum"),
        denominator=("den", "sum"),
        signal_count=("factor", "size")
    )
)

article_scores["article_score"] = (
    article_scores["numerator"]
    / article_scores["denominator"]
)


# ---------------------------------------------------------
# 3. 再计算 D、R 总体得分
# ---------------------------------------------------------

factor_scores = (
    article_scores.groupby(
        "factor",
        as_index=False
    )
    .agg(
        score=("article_score", "mean"),
        article_count=("title", "count"),
        min_score=("article_score", "min"),
        max_score=("article_score", "max")
    )
)


# ---------------------------------------------------------
# 4. 顺便计算每个月的趋势
# ---------------------------------------------------------

article_scores["date"] = pd.to_datetime(
    article_scores["date"]
)

article_scores["month"] = (
    article_scores["date"]
    .dt.to_period("M")
    .astype(str)
)

monthly_scores = (
    article_scores.groupby(
        ["month", "factor"],
        as_index=False
    )
    .agg(
        score=("article_score", "mean"),
        article_count=("title", "count")
    )
)


# ---------------------------------------------------------
# 5. 输出
# ---------------------------------------------------------

print("\n========== 文章级评分 ==========")
print(
    article_scores[
        [
            "date",
            "title",
            "factor",
            "signal_count",
            "article_score"
        ]
    ].to_string(index=False)
)

print("\n========== 总体因子评分 ==========")

for _, row in factor_scores.iterrows():

    print(
        f"{row['factor']}："
        f"S = {row['score']:.4f} | "
        f"文章数 = {int(row['article_count'])}"
    )

print("\n========== 月度趋势 ==========")
print(monthly_scores.to_string(index=False))


# 保存
article_scores.to_csv(
    "article_factor_scores.csv",
    index=False,
    encoding="utf-8-sig"
)

factor_scores.to_csv(
    "factor_scores.csv",
    index=False,
    encoding="utf-8-sig"
)

monthly_scores.to_csv(
    "monthly_factor_scores.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n已生成：")
print("article_factor_scores.csv")
print("factor_scores.csv")
print("monthly_factor_scores.csv")