import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import gc
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 加载数据
# ==========================================
print("🚀 正在加载特征数据...")
DATA_DIR = 'processed_data'
X_train_all = pd.read_pickle(os.path.join(DATA_DIR, 'X_train.pkl'))
y_train_all = pd.read_pickle(os.path.join(DATA_DIR, 'y_train.pkl'))
X_test = pd.read_pickle(os.path.join(DATA_DIR, 'X_test.pkl'))

# ==========================================
# 2. 特征清洗 (保持严谨)
# ==========================================
print("🧹 剔除短期记忆特征 (防止泄露)...")
# 我们保留 rolling_mean_30 试试？不，为了稳健，还是删掉，靠数据量取胜
toxic_features = [
    'sales_lag_1', 
    'sales_lag_7', 
    'sales_lag_14', 
    'rolling_mean_7', 
    'zero_sales_freq',      
    'consecutive_zero_days' 
]
# 注意：我这次没删 'rolling_mean_30'，我们稍微冒一点险，因为它包含了长期趋势
# 如果效果不好，下次再把 rolling_mean_30 也删掉
cols_to_drop = [c for c in toxic_features if c in X_train_all.columns]
X_train_all = X_train_all.drop(columns=cols_to_drop)
X_test = X_test.drop(columns=cols_to_drop)
print(f"✅ 已剔除: {cols_to_drop}")

# ==========================================
# 3. 优化策略：扩充训练数据 (Boost Data)
# ==========================================
# 之前只用了 2017，现在我们回溯到 2016-06-01
# (避开了2016-04的地震，但增加了一年的数据量)
if 'date' not in X_train_all.columns:
    # 重构日期用于筛选
    X_train_all['date'] = pd.to_datetime(dict(year=X_train_all.year, month=X_train_all.month, day=X_train_all.day))

print("📈 正在扩充训练集 (2016-06-01 ~ 2017-08-15)...")
mask_boost = X_train_all['date'] >= '2016-06-01'

X_train_all = X_train_all[mask_boost].drop(columns=['date']) # 筛选完记得扔掉 date
y_train_all = y_train_all[mask_boost]

print(f"✅ 新训练集规模: {X_train_all.shape} (数据量翻倍！)")

# ==========================================
# 4. 准备训练
# ==========================================
y_train_log_all = np.log1p(y_train_all)

# 切分验证集 (最后16天)
VAL_SIZE = 30000
X_train = X_train_all.iloc[:-VAL_SIZE]
y_train_log = y_train_log_all.iloc[:-VAL_SIZE]
X_val = X_train_all.iloc[-VAL_SIZE:]
y_val_log = y_train_log_all.iloc[-VAL_SIZE:]

# Target Encoding
group_cols = ['store_nbr', 'family']
temp_df = X_train.copy()
temp_df['target'] = y_train_log 
target_map = temp_df.groupby(group_cols)['target'].mean().to_dict()
global_mean = y_train_log.mean()

def apply_target_encoding(df, mapping, cols, default_val):
    return df.set_index(cols).index.map(mapping).fillna(default_val).values

X_train['store_family_mean'] = apply_target_encoding(X_train, target_map, group_cols, global_mean)
X_val['store_family_mean'] = apply_target_encoding(X_val, target_map, group_cols, global_mean)
X_test['store_family_mean'] = apply_target_encoding(X_test, target_map, group_cols, global_mean)

# ==========================================
# 5. 训练模型 (参数增强)
# ==========================================
print("🔥 开始训练 (增强模式)...")
cat_features = ['store_nbr', 'family', 'city', 'state', 'store_type']
cat_features = [c for c in cat_features if c in X_train.columns]

train_data = lgb.Dataset(X_train, label=y_train_log, categorical_feature=cat_features)
val_data = lgb.Dataset(X_val, label=y_val_log, reference=train_data, categorical_feature=cat_features)

# 🚀 优化后的参数：更聪明，更强壮
params = {
    'objective': 'regression', 
    'metric': 'rmse', 
    'boosting_type': 'gbdt',
    
    'num_leaves': 31,         # 回调到标准值 (模型容量变大)
    'learning_rate': 0.03,    # 稍微学快一点
    'feature_fraction': 0.8,  # 使用更多特征
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    
    'lambda_l1': 0.0,         # 减少正则化，让模型放开手脚
    'lambda_l2': 0.0,
    
    'verbose': -1, 
    'seed': 42, 
    'force_row_wise': True
}

model = lgb.train(
    params, train_data, valid_sets=[train_data, val_data], num_boost_round=3000,
    callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=200)]
)

# ==========================================
# 6. 预测与对齐
# ==========================================
print("🔮 预测中...")
preds_log = model.predict(X_test, num_iteration=model.best_iteration)
preds = np.expm1(preds_log)
preds = np.maximum(preds, 0)

print("🔧 安全对齐 ID...")
X_test_with_pred = X_test.copy()
X_test_with_pred['sales_pred'] = preds
X_test_with_pred['date'] = pd.to_datetime(dict(year=X_test_with_pred.year, month=X_test_with_pred.month, day=X_test_with_pred.day))

raw_test = pd.read_csv('raw data/test.csv')
raw_test['date'] = pd.to_datetime(raw_test['date'])

le = LabelEncoder()
le.fit(raw_test['family'])
raw_test['family_int'] = le.transform(raw_test['family'])

submission = raw_test.merge(
    X_test_with_pred[['store_nbr', 'family', 'date', 'sales_pred']], 
    left_on=['store_nbr', 'family_int', 'date'],
    right_on=['store_nbr', 'family', 'date'],
    how='left'
)

final_sub = pd.DataFrame({'id': submission['id'], 'sales': submission['sales_pred']})
final_sub.to_csv('submission_boost.csv', index=False)
print("🎉 增强版文件: submission_boost.csv")