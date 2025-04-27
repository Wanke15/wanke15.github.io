# 1. 加载原始数据
# 2. 构建特征：时间、滞后
# 3. xgboost模型训练
# 4. 训练集和验证集损失可视化
# 5. 未来数据预测
# 6. 特征重要性可视化

import json
from collections import defaultdict

import time
import datetime

import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tqdm import tqdm

tqdm.pandas()

print('开始读取数据...', datetime.datetime.now())
df = pd.read_csv(r"../data/sf_order_df_20240101_20250430.csv")
'''
# 字段名称
biz_date,prod_id,prod_name,branch_id,branch_name,sum_real_price,mean_real_price,sum_sale_price,mean_sale_price,sum_sku_cnt,sum_order_cnt,mean_sku_real_price,mean_sku_sale_price,mean_sku_price_discount
'''
print('数据读取完毕！', datetime.datetime.now())

df["date"] = pd.to_datetime(df["biz_date"])
df["item_id"] = df["prod_id"].astype("str").fillna("unk_prod_id")
print(df.columns)

# 只取top_n数据量比较多的商品
top_n = 5000
df = df[df['item_id'].isin(df['item_id'].value_counts().index.tolist()[0:top_n])]
print(df['item_id'].value_counts().head(top_n))
print(df['branch_name'].value_counts().head(top_n))
# df = df[df['prod_id'] == xxxx]

train_end_date = '2025-03-01'

# df["sum_sku_cnt"] = np.log(3 + df["sum_sku_cnt"])

prod_value_cnts = df['item_id'].value_counts().to_dict()
prod_branch_map = dict(zip(df['item_id'].tolist(), df['branch_name'].tolist()))
prod_name_map = dict(zip(df['item_id'].tolist(), df['prod_name'].tolist()))

df["sales"] = df["sum_sku_cnt"]

# 1. 构造完整日期表
all_dates = pd.date_range(df["date"].min(), df["date"].max())
all_items = df["item_id"].unique()
full_index = pd.MultiIndex.from_product([all_items, all_dates], names=["item_id", "date"])
full_df = pd.DataFrame(index=full_index).reset_index()

# 2. 合并原始数据，补全缺失日期
df_full = pd.merge(full_df, df, on=["item_id", "date"], how="left")
df_full["is_filled"] = df_full["sales"].isna().astype(int)

# 3. 对缺失值做填充
df_full["prod_name"] = df_full["item_id"].map(prod_name_map)
df_full["branch_name"] = df_full["item_id"].map(prod_branch_map)

df_full["sales"] = df_full["sales"].fillna(0)
df_full["prod_avg_real_price"] = df_full["mean_sku_real_price"].fillna(method="ffill")  # 或填均值
df_full["prod_avg_sale_price"] = df_full["mean_sku_sale_price"].fillna(method="ffill")  # 或填均值
df_full["mean_sku_price_diff"] = df_full["prod_avg_sale_price"] - df_full["prod_avg_real_price"]

df = df_full.sort_values(["item_id", "date"])

# 时间特征
df['month_of_year'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['day_of_month'] = df['date'].dt.day
df['day_of_year'] = df['date'].dt.dayofyear
df['is_weekend'] = (df.date.dt.dayofweek >= 5).astype(int)

lag_feature_cols = []
# for d in [1, 7, 14, 30, 60]:
# 滞后特征
for d in range(1, 30):
    df[f'sale_lag_{d}'] = df.groupby('item_id')['sales'].shift(d)
    lag_feature_cols.append(f'sale_lag_{d}')

def _max_current_interval(x):
    '''最高值到当前周期的距离'''
    return len(x) - np.argmax(x[::-1]) - 1 if not np.isnan(x).all() else np.nan

# 滞后窗口特征：加和、均值、最大值间隔
for d in tqdm([7, 14, 30, 60], desc='lag window features'):
    df[f'sale_lag_{d}_sum'] = df.groupby('item_id')['sales'].shift(d).rolling(window=d).sum()
    df[f'sale_lag_{d}_mean'] = df.groupby('item_id')['sales'].shift(d).rolling(window=d).mean()

    df[f'sale_lag_{d}_sum_1'] = df.groupby('item_id')['sales'].rolling(window=d).sum().reset_index(level=0,
                                                                                                   drop=True).reindex(
        df.index)
    df[f'sale_lag_{d}_mean_1'] = df.groupby('item_id')['sales'].rolling(window=d).mean().reset_index(level=0,
                                                                                                     drop=True).reindex(
        df.index)

    df[f'sale_lag_{d}_max_interval'] = df.groupby('item_id')['sales'].rolling(d).apply(_max_current_interval,
                                                                                       raw=True).reset_index(level=0,
                                                                                                             drop=True).reindex(
        df.index)
    df[f'sale_lag_{d}_max_interval_1'] = df.groupby('item_id')['sales'].shift(1).rolling(d).apply(_max_current_interval,
                                                                                                  raw=True)

    lag_feature_cols.append(f'sale_lag_{d}_sum')
    lag_feature_cols.append(f'sale_lag_{d}_mean')

    lag_feature_cols.append(f'sale_lag_{d}_max_interval')
    lag_feature_cols.append(f'sale_lag_{d}_max_interval_1')

    lag_feature_cols.append(f'sale_lag_{d}_sum_1')
    lag_feature_cols.append(f'sale_lag_{d}_mean_1')

# 窗口内无销量统计特征
# for d in [7, 14, 30, 60]:
#     df[f'no_sale_days_lag_{d}_sum'] = df.groupby('item_id')['is_filled'].shift(d).rolling(window=d).sum()
#     df[f'no_sale_days_lag_{d}_mean'] = df.groupby('item_id')['is_filled'].shift(d).rolling(window=d).mean()
#     df[f'no_sale_days_lag_{d}_std'] = df.groupby('item_id')['is_filled'].shift(d).rolling(window=d).std()

#     lag_feature_cols.append(f'no_sale_days_lag_{d}_sum')
#     lag_feature_cols.append(f'no_sale_days_lag_{d}_mean')
#     lag_feature_cols.append(f'no_sale_days_lag_{d}_std')


# 创建标签：未来销量（30天、1天）
df['label'] = df.groupby('item_id')['sales'].transform(lambda x: x.shift(-30).rolling(window=30).sum())
# df['label'] = df.groupby('item_id')['sales'].transform(lambda x: x.shift(-1))

# 类别特征转换
cat_features = ['item_id', 'branch_name', 'prod_name']
for cat in cat_features:
    df[cat] = df[cat].fillna(f"unk_{cat}").astype("category")

# 删除没有标签的数据
future_df = df[df.date >= train_end_date].copy()
print(future_df.tail())

df = df.dropna(subset=['label'])
df['gt'] = df['label']

# 准备训练数据
features_cols = cat_features + [
    'month_of_year', 'day_of_week', 'day_of_month', 'day_of_year', 'is_weekend',
    'prod_avg_real_price', 'prod_avg_sale_price',
    'mean_sku_price_discount', 'mean_sku_price_diff',
] + lag_feature_cols

features_cols.append("date")
print('特征名称：', features_cols)


# 评测指标
def result_score(pred, gt):
    pred_residual = gt - np.abs(np.round(pred) - gt)
    pred_residual[pred_residual < 0] = 0
    score = (pred_residual / gt.values).mean()
    return score


def evaluate(x, y, flag):
    y_pred = model.predict(xgb.DMatrix(x.drop(['date'], axis=1), enable_categorical=True))
    # y_pred = model.predict(x.drop(['date'], axis=1))

    y_pred_list = y_pred.tolist()

    mse = metrics.mean_squared_error(y, y_pred_list)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y, y_pred_list)
    r2 = metrics.r2_score(y, y_pred_list)

    # print("均方误差 (MSE):", mse)
    # print("均方根误差 (RMSE):", rmse)
    print("平均绝对误差 (MAE):", mae)
    print("拟合优度 (R-squared):", r2)
    biz_acc = result_score(y_pred, y)
    print("业务准确率：", biz_acc)

cur_prod_df = df

X_train, X_test = cur_prod_df[cur_prod_df.date < train_end_date][features_cols], \
cur_prod_df[cur_prod_df.date >= train_end_date][features_cols]
y_train, y_test = cur_prod_df[cur_prod_df.date < train_end_date]['label'], \
cur_prod_df[cur_prod_df.date >= train_end_date]['label']

# 模型训练
target = "label"

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 设置参数
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.2,
    'max_depth': 6,
    'enable_categorical': True,
    'seed': 101,
}
# 训练模型
start = time.time()

evals_result = {}
dtrain = xgb.DMatrix(data=X_train.drop(['date'], axis=1), label=y_train, enable_categorical=True)
dval = xgb.DMatrix(data=X_val.drop(['date'], axis=1), label=y_val, enable_categorical=True)
model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=1000,
    evals=[(dtrain, 'train'), (dval, 'valid')],
    early_stopping_rounds=100,
    evals_result=evals_result,
    verbose_eval=1
)
print('训练耗时：', time.time() - start)

# 可视化训练和验证集上的loss
epochs = len(evals_result['train']['rmse'])
x_axis = range(0, epochs)
plt.figure(figsize=(15, 10))
plt.plot(x_axis, evals_result['train']['rmse'], label='Train')
plt.plot(x_axis, evals_result['valid']['rmse'], label='Validation')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.title('XGBoost RMSE')
plt.legend()
plt.tight_layout()
plt.savefig("train_vis_xgb_dmat_v3.png")
# plt.show()

model.save_model(f"./save_model_xgb.json")

print(f"\n训练集指标：" + "#" * 50)
evaluate(X_train, y_train, "train")

print(f"\n验证集指标：" + "#" * 50)
evaluate(X_val, y_val, "val")

print(f"\n测试集指标：" + "#" * 50)
evaluate(X_test, y_test, "test")

# X_predict_df = X_test[X_test["date"] == X_test["date"].max()]

X_predict_df = future_df
X_predict_df["pred"] = np.round(model.predict(xgb.DMatrix(X_predict_df[features_cols].drop(['date'], axis=1), enable_categorical=True)))

# # 保存预测结果
X_predict_df = X_predict_df[['date', 'item_id', 'prod_name', 'branch_name', 'label', 'pred']]
# X_predict_df['label'] = np.e ** (X_predict_df['label']) - 3
# X_predict_df['pred'] = np.e ** (X_predict_df['pred']) - 3

X_predict_df.to_csv("future_30d_sales_predictions_xgb_dmat.csv", index=False)
print("预测完成，结果保存在 future_30d_sales_predictions_xgb_dmat.csv")

feature_importance = model.get_score(importance_type='weight')
feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

plt.figure(figsize=(15, 10))
plt.barh([x[0] for x in feature_importance], [x[1] for x in feature_importance], color=mcolors.TABLEAU_COLORS)
plt.title("Tree Model Feature Importances")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance_xgb_dmat_v3.png")
# plt.show()

print(feature_importance)

