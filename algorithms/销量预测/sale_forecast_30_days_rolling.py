import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# （1）构建特征
# （2）预测。并把预测值添加到特征表，基于最新的预测值更新特征，如lag特征
# （3）循环步骤2，直到最大预测日期

class Sales30DayForecaster:
    def __init__(self, max_lag=30):
        self.model = None
        self.max_lag = max_lag
        self.features = []

    def _add_time_features(self, df):
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['is_weekend'] = df['day_of_week'] >= 5
        return df

    def _add_campaign_features(self, df, promo_df):
        return df.merge(promo_df, on=['sku_id', 'date'], how='left')

    def _add_lag_features(self, df_pred, df_hist):
        df_result = df_pred.copy()
        for lag in [3, 7, 14]:
            lag_col = f'sales_lag_{lag}'
            lag_vals = (
                df_hist.groupby('sku_id')['sales']
                .apply(lambda x: x.iloc[-lag] if len(x) >= lag else np.nan)
                .reset_index().rename(columns={'sales': lag_col})
            )
            df_result = df_result.merge(lag_vals, on='sku_id', how='left')

        for window in [7, 14, 30]:
            roll_col = f'sales_roll_mean_{window}'
            roll_vals = (
                df_hist.groupby('sku_id')['sales']
                .apply(lambda x: x.tail(window).mean() if len(x) >= window else np.nan)
                .reset_index().rename(columns={'sales': roll_col})
            )
            df_result = df_result.merge(roll_vals, on='sku_id', how='left')
        return df_result

    def _generate_training_data(self, df_all, item_df, promo_df):
        df = df_all.sort_values(by=['sku_id', 'date']).copy()
        df['target_30d_sales'] = (
            df.groupby('sku_id')['sales']
            .transform(lambda x: x.shift(-1).rolling(30).sum())
        )

        feature_rows = []
        for date in df['date'].unique():
            hist = df[(df['date'] < date) & (df['date'] >= date - pd.Timedelta(days=self.max_lag))]
            current = df[df['date'] == date][['sku_id', 'date', 'target_30d_sales']].copy()
            current = current.merge(item_df, on='sku_id', how='left')
            current = self._add_time_features(current)
            current = self._add_campaign_features(current, promo_df)
            current = self._add_lag_features(current, hist)
            feature_rows.append(current)

        train_df = pd.concat(feature_rows, ignore_index=True).dropna(subset=['target_30d_sales'])
        return train_df

    def train(self, df_all, item_df, promo_df):
        train_df = self._generate_training_data(df_all, item_df, promo_df)
        self.features = [col for col in train_df.columns if col not in ['date', 'sales', 'target_30d_sales']]
        X = train_df[self.features]
        y = train_df['target_30d_sales']
        self.model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, objective='reg:squarederror')
        self.model.fit(X, y)

    def predict_single_day(self, df_history, item_df, promo_df, predict_date):
        recent_hist = df_history[
            (df_history['date'] < predict_date) &
            (df_history['date'] >= predict_date - pd.Timedelta(days=self.max_lag))
        ]
        recent_skus = df_history[df_history['date'] == df_history['date'].max()]['sku_id'].unique()
        df_pred = pd.DataFrame({'sku_id': recent_skus})
        df_pred['date'] = predict_date
        df_pred = df_pred.merge(item_df, on='sku_id', how='left')
        df_pred = self._add_time_features(df_pred)
        df_pred = self._add_campaign_features(df_pred, promo_df)
        df_pred = self._add_lag_features(df_pred, recent_hist)
        X_pred = df_pred[self.features].dropna()
        df_pred = df_pred.loc[X_pred.index]
        df_pred['predicted_30d_sales'] = self.model.predict(X_pred)
        return df_pred[['sku_id', 'date', 'predicted_30d_sales']]

    def rolling_predict(self, df_history, item_df, promo_df, start_date, end_date):
        all_preds = []
        curr_date = start_date
        while curr_date <= end_date:
            preds = self.predict_single_day(df_history, item_df, promo_df, curr_date)
            all_preds.append(preds)
            curr_date += pd.Timedelta(days=1)
        return pd.concat(all_preds, ignore_index=True)
