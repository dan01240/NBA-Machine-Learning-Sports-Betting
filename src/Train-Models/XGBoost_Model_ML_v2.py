import sqlite3
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, calibration_curve
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time

# タイムスタンプを生成（ファイル名用）
timestamp = int(time.time())

# データの読み込み
dataset = "dataset_2019-25"
con = sqlite3.connect("../../Data/dataset.sqlite")
data = pd.read_sql_query(f"select * from \"{dataset}\"", con, index_col="index")
con.close()

margin = data['Home-Team-Win']
data.drop(['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU'],
          axis=1, inplace=True)

data = data.values
data = data.astype(float)

# キャリブレーション誤差を計算する関数
def calculate_calibration_error(model, X, y, n_bins=10):
    # 予測
    dmatrix = xgb.DMatrix(X)
    y_pred = model.predict(dmatrix)
    
    # 各サンプルのクラス1の予測確率を取得
    if len(y_pred.shape) > 1:
        y_pred_proba = y_pred[:, 1]
    else:
        try:
            y_pred_proba = np.array([p[1] if isinstance(p, np.ndarray) else p for p in y_pred])
        except:
            y_pred_proba = y_pred
    
    # キャリブレーション曲線を計算
    prob_true, prob_pred = calibration_curve(y, y_pred_proba, n_bins=n_bins, strategy='quantile')
    
    # キャリブレーション誤差の計算
    calibration_error = np.mean(np.abs(prob_true - prob_pred))
    
    return calibration_error

# モデルトレーニングと評価
acc_results = []
cal_results = []  # キャリブレーション誤差の結果
best_cal_error = float('inf')  # 最良のキャリブレーション誤差

print("モデル学習開始 - キャリブレーションエラーを指標に最適化しています...")

for x in tqdm(range(300)):
    # データ分割
    x_train, x_test, y_train, y_test = train_test_split(data, margin, test_size=.2)

    train = xgb.DMatrix(x_train, label=y_train)
    test = xgb.DMatrix(x_test, label=y_test)

    # モデルパラメータ
    param = {
        'max_depth': 3,
        'eta': 0.01,
        'objective': 'multi:softprob',
        'num_class': 2,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    epochs = 750

    # モデル学習
    model = xgb.train(param, train, epochs)
    
    # 予測と精度評価
    predictions = model.predict(test)
    y_pred = np.argmax(predictions, axis=1)
    acc = round(accuracy_score(y_test, y_pred) * 100, 1)
    
    # キャリブレーション誤差の計算
    cal_error = calculate_calibration_error(model, x_test, y_test)
    cal_error_pct = round(cal_error * 100, 2)
    
    print(f"精度: {acc}%, キャリブレーションエラー: {cal_error_pct}%")
    
    acc_results.append(acc)
    cal_results.append(cal_error)
    
    # キャリブレーションエラーが最小のモデルを保存
    if cal_error < best_cal_error:
        best_cal_error = cal_error
        model_path = f'../../Models/XGBoost_CAL{cal_error_pct}%_ACC{acc}%_ML-{timestamp}.json'
        model.save_model(model_path)
        print(f"★ 新しい最良モデル保存: {model_path} (キャリブレーションエラー: {cal_error_pct}%)")

# 結果を表示
best_idx = np.argmin(cal_results)
print("\n=== トレーニング結果 ===")
print(f"最良のキャリブレーションエラー: {cal_results[best_idx]*100:.2f}%")
print(f"対応する精度: {acc_results[best_idx]}%")
print(f"全体の平均精度: {np.mean(acc_results):.1f}%")
print(f"全体の平均キャリブレーションエラー: {np.mean(cal_results)*100:.2f}%")