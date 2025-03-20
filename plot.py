import sqlite3
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    roc_auc_score, 
    precision_recall_curve, 
    average_precision_score
)
from sklearn.calibration import calibration_curve

class ModelEvaluator:
    def __init__(self, model_path, dataset_name="dataset_2019-25"):
        """
        モデル評価クラス
        
        Args:
            model_path (str): 評価するモデルのパス
            dataset_name (str): 使用するデータセット名
        """
        self.model_path = model_path
        self.dataset_name = dataset_name
        self.model = None
        self.X = None
        self.y = None
    
    def load_data(self):
        """データベースからデータを読み込む"""
        con = sqlite3.connect("./Data/dataset.sqlite")
        data = pd.read_sql_query(f"select * from \"{self.dataset_name}\"", con, index_col="index")
        con.close()
        
        # ターゲット変数と特徴量の分離
        self.y = data['Home-Team-Win']
        X = data.drop([
            'Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 
            'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU'
        ], axis=1)
        
        # NumPy配列に変換
        self.X = X.values.astype(float)
        self.y = self.y.values
    
    def load_model(self):
        """モデルをロード"""
        self.model = xgb.Booster()
        self.model.load_model(self.model_path)
    
    def comprehensive_evaluation(self):
        """
        包括的なモデル評価を実行
        """
        # データとモデルのロード
        self.load_data()
        self.load_model()
        
        # データ分割
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # XGBoostデータマトリックスの作成
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # 予測
        y_pred_proba = self.model.predict(dtest)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # 分類レポート
        print("=== 分類レポート ===")
        print(classification_report(y_test, y_pred))
        
        # 混同行列
        print("\n=== 混同行列 ===")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('混同行列')
        plt.colorbar()
        plt.xlabel('予測されたラベル')
        plt.ylabel('真のラベル')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        # ROC曲線
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲線 (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('偽陽性率')
        plt.ylabel('真陽性率')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve.png')
        plt.close()
        
        # Precision-Recall曲線
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall曲線 (平均精度 = {avg_precision:.2f})')
        plt.savefig('precision_recall_curve.png')
        plt.close()
        
        # キャリブレーション曲線
        prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
        
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker='o', label='モデルのキャリブレーション')
        plt.plot([0, 1], [0, 1], linestyle='--', label='理想的なキャリブレーション')
        plt.xlabel('予測確率')
        plt.ylabel('真の確率')
        plt.title('キャリブレーション曲線')
        plt.legend()
        plt.savefig('detailed_calibration_curve.png')
        plt.close()
        
        # 予測確率の分布
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.hist(y_pred_proba[y_test == 0], bins=30, alpha=0.5, label='負のクラス')
        plt.hist(y_pred_proba[y_test == 1], bins=30, alpha=0.5, label='正のクラス')
        plt.title('予測確率の分布')
        plt.xlabel('予測確率')
        plt.ylabel('頻度')
        plt.legend()
        
        plt.subplot(122)
        plt.boxplot([y_pred_proba[y_test == 0], y_pred_proba[y_test == 1]], labels=['負のクラス', '正のクラス'])
        plt.title('予測確率のボックスプロット')
        plt.ylabel('予測確率')
        
        plt.tight_layout()
        plt.savefig('probability_distribution.png')
        plt.close()

def main():
    # モデルパスは、トレーニング時に生成されたモデルのパスに置き換えてください
    model_path = './Models/CAL/XGBoost_CAL1.31%_ACC61.7%_ML-1742435642.json'
    evaluator = ModelEvaluator(model_path)
    evaluator.comprehensive_evaluation()

if __name__ == "__main__":
    main()