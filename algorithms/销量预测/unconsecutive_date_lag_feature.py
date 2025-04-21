import pandas as pd

# （1）构造所有日期和所有商品的笛卡尔积
# （2）关联原始表，填充缺失值

df = pd.read_csv("sample_your_data.csv", parse_dates=["date"])
print(df)

# 1. 构造完整日期表
all_dates = pd.date_range(df["date"].min(), df["date"].max())
all_items = df["item_id"].unique()
full_index = pd.MultiIndex.from_product([all_items, all_dates], names=["item_id", "date"])
full_df = pd.DataFrame(index=full_index).reset_index()

# 2. 合并原始数据，补全缺失日期
df_full = pd.merge(full_df, df, on=["item_id", "date"], how="left")
df_full["is_filled"] = df_full["sales"].isna().astype(int)

# 3. 对缺失值做填充
df_full["sales"] = df_full["sales"].fillna(0)
df_full["views"] = df_full["views"].fillna(0)
df_full["discount"] = df_full["discount"].fillna(1.0)
df_full["avg_price"] = df_full["avg_price"].fillna(method="ffill")  # 或填均值
df_full["promotion_type"] = df_full["promotion_type"].fillna("none")

df = df_full.sort_values(["item_id", "date"])
print(df)
