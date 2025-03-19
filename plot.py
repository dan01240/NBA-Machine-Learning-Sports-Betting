import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
import xgboost as xgb
import sqlite3
import pandas as pd

# データの読み込み（必要に応じて変更）
def load_data():
    # データをロード
    dataset = "dataset_2012-24_new"
    con = sqlite3.connect("Data/dataset.sqlite")
    data = pd.read_sql_query(f"select * from \"{dataset}\"", con, index_col="index")
    con.close()
    
    # 目標値と特徴量を分離
    margin = data['Home-Team-Win']
    data.drop(['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU'],
              axis=1, inplace=True)
    
    return data.values, margin

# モデルの読み込み
def load_model():
    # XGBoostのモデルをロード
    model = xgb.Booster()
    model.load_model('Models/XGBoost_Models/XGBoost_68.7%_ML-4.json')
    return model

# キャリブレーションプロットの作成
def create_calibration_plot(model, X, y):
    # テストデータに対する予測
    dmatrix = xgb.DMatrix(X)
    y_pred = model.predict(dmatrix)
    
    # 各サンプルの1クラス（ホームチーム勝利）の予測確率を取得
    y_pred_proba = y_pred[:, 1] if len(y_pred.shape) > 1 else np.array([p[1] for p in y_pred])
    
    # キャリブレーション曲線を計算
    prob_true, prob_pred = calibration_curve(y, y_pred_proba, n_bins=10, strategy='quantile')
    
    # プロットの作成
    plt.figure(figsize=(10, 8))
    
    # 完璧なキャリブレーション（対角線）
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated', color='orange', dashes=(5, 5), linewidth=2)
    
    # 実際のキャリブレーション曲線
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Reliability Curve', color='#1f77b4')
    
    # グラフのタイトルと軸ラベル
    plt.title('Reliability Plot (Calibration Plot)', fontsize=20)
    plt.xlabel('Mean Predicted Probability', fontsize=14)
    plt.ylabel('True Probability', fontsize=14)
    
    # 凡例
    plt.legend(fontsize=12)
    
    # グリッド線の追加
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 軸の範囲を設定
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    
    # プロットを保存
    plt.savefig('reliability_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # キャリブレーション誤差の計算
    calibration_error = np.mean(np.abs(prob_true - prob_pred))
    print(f"Average calibration error: {calibration_error:.4f} or {calibration_error*100:.2f}%")
    
    return calibration_error

# メイン処理
def main():
    X, y = load_data()
    model = load_model()
    
    # キャリブレーションプロットの作成
    calibration_error = create_calibration_plot(model, X, y)
    
    return calibration_error

if __name__ == "__main__":
    main()