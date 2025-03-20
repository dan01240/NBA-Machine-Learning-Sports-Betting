import sqlite3
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, 
    brier_score_loss, 
    log_loss, 
    roc_auc_score,
    precision_score, 
    recall_score, 
    f1_score
)
from sklearn.calibration import calibration_curve, CalibrationDisplay

class XGBoostModelTrainer:
    def __init__(self, dataset_name="dataset_2019-25", random_state=42):
        """
        XGBoostモデルのトレーニングと評価を行うクラス
        
        Args:
            dataset_name (str): 使用するデータセット名
            random_state (int): 再現性のための乱数シード
        """
        self.dataset_name = dataset_name
        self.random_state = random_state
        self.data = None
        self.X = None
        self.y = None
        
        # 結果を保存するリスト
        self.acc_results = []
        self.cal_results = []
        self.auc_results = []
        self.brier_results = []
        
        # 最良のモデル情報
        self.best_model = None
        self.best_cal_error = float('inf')
        
    def load_data(self):
        """データベースからデータを読み込む"""
        con = sqlite3.connect("../../Data/dataset.sqlite")
        self.data = pd.read_sql_query(f"select * from \"{self.dataset_name}\"", con, index_col="index")
        con.close()
        
        # ターゲット変数と特徴量の分離
        self.y = self.data['Home-Team-Win']
        self.X = self.data.drop([
            'Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 
            'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU'
        ], axis=1)
        
        # NumPy配列に変換
        self.X = self.X.values.astype(float)
        self.y = self.y.values
        
    def calculate_calibration_metrics(self, y_true, y_pred_proba):
        """
        キャリブレーション関連の指標を計算
        
        Args:
            y_true (array): 真のラベル
            y_pred_proba (array): 予測確率
        
        Returns:
            dict: キャリブレーション指標
        """
        # キャリブレーション曲線
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
        calibration_error = np.mean(np.abs(prob_true - prob_pred))
        
        # 追加の指標
        brier_score = brier_score_loss(y_true, y_pred_proba)
        log_loss_score = log_loss(y_true, y_pred_proba)
        
        return {
            'calibration_error': calibration_error,
            'brier_score': brier_score,
            'log_loss': log_loss_score
        }
    
    def plot_calibration(self, y_true, y_pred_proba, metrics, iteration):
        """
        キャリブレーションプロットを作成
        
        Args:
            y_true (array): 真のラベル
            y_pred_proba (array): 予測確率
            metrics (dict): キャリブレーション指標
            iteration (int): 現在のイテレーション
        """
        plt.figure(figsize=(12, 5))
        
        # キャリブレーション曲線
        plt.subplot(121)
        CalibrationDisplay.from_predictions(
            y_true, y_pred_proba, 
            n_bins=10, 
            name=f'Iteration {iteration}',
            ax=plt.gca()
        )
        plt.title(f'Calibration Curve\nCal Error: {metrics["calibration_error"]:.4f}')
        
        # 予測確率ヒストグラム
        plt.subplot(122)
        plt.hist([y_pred_proba[y_true == 0], y_pred_proba[y_true == 1]], 
                 label=['Negative', 'Positive'], 
                 bins=30, alpha=0.7)
        plt.title('Predicted Probability Distribution')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'calibration_plot_iter_{iteration}.png')
        plt.close()
    
    def train_and_evaluate(self, num_iterations=1000):
        """
        モデルのトレーニングと評価を実行
        
        Args:
            num_iterations (int): 学習の繰り返し回数
        """
        # データ読み込み
        self.load_data()
        
        print("モデル学習開始 - キャリブレーションエラーを指標に最適化...")
        
        # タイムスタンプ生成
        timestamp = int(time.time())
        
        for iteration in tqdm(range(num_iterations)):
            # 層化抽出によるデータ分割
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, 
                test_size=0.2, 
                random_state=self.random_state + iteration, 
                stratify=self.y
            )
            
            # XGBoostデータマトリックスの作成
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            
            # モデルパラメータ
            param = {
                'max_depth': 3,
                'eta': 0.01,
                'objective': 'binary:logistic',  # 2値分類
                'eval_metric': 'logloss',
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'alpha': 0.1,  # L1正則化
                'lambda': 0.1,  # L2正則化
            }
            
            # モデルトレーニング
            model = xgb.train(
                param, 
                dtrain, 
                num_boost_round=750,
                evals=[(dtest, 'eval')],
                early_stopping_rounds=50,
                verbose_eval=False
            )
            
            # テストデータでの予測
            y_pred_proba = model.predict(dtest)
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            # 評価指標の計算
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            # キャリブレーション指標の計算
            cal_metrics = self.calculate_calibration_metrics(y_test, y_pred_proba)
            
            # 結果の記録
            self.acc_results.append(acc)
            self.cal_results.append(cal_metrics['calibration_error'])
            self.auc_results.append(auc)
            self.brier_results.append(cal_metrics['brier_score'])
            
            # キャリブレーションエラーが最小のモデルを保存
            if cal_metrics['calibration_error'] < self.best_cal_error:
                self.best_cal_error = cal_metrics['calibration_error']
                self.best_model = model
                
                # モデル保存
                model_path = f'../../Models/XGBoost_CAL{cal_metrics["calibration_error"]*100:.2f}%_ACC{acc*100:.1f}%_ML-{timestamp}.json'
                model.save_model(model_path)
                
                # キャリブレーションプロットの作成
                self.plot_calibration(y_test, y_pred_proba, cal_metrics, iteration)
                
                print(f"\n★ 新しい最良モデル保存: {model_path}")
                print(f"キャリブレーションエラー: {cal_metrics['calibration_error']:.4f}")
                print(f"精度: {acc:.4f}")
                print(f"AUC: {auc:.4f}")
        
        # 最終結果の表示
        self.print_summary_results()
    
    def print_summary_results(self):
        """
        学習結果の概要を表示
        """
        best_idx = np.argmin(self.cal_results)
        
        print("\n=== トレーニング結果 ===")
        print(f"最良のキャリブレーションエラー: {self.cal_results[best_idx]:.4f}")
        print(f"対応する精度: {self.acc_results[best_idx]:.4f}")
        print(f"対応するAUC: {self.auc_results[best_idx]:.4f}")
        
        print("\n平均指標:")
        print(f"平均精度: {np.mean(self.acc_results):.4f} ± {np.std(self.acc_results):.4f}")
        print(f"平均キャリブレーションエラー: {np.mean(self.cal_results):.4f} ± {np.std(self.cal_results):.4f}")
        print(f"平均AUC: {np.mean(self.auc_results):.4f} ± {np.std(self.auc_results):.4f}")
        print(f"平均ブライアースコア: {np.mean(self.brier_results):.4f} ± {np.std(self.brier_results):.4f}")

def main():
    trainer = XGBoostModelTrainer()
    trainer.train_and_evaluate()

if __name__ == "__main__":
    main()