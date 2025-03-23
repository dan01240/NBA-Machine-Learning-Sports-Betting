import sqlite3
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm
import random
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
    def __init__(self, dataset_name="dataset_2012-25_test", random_state=42, holdout_size=0.2):
        """
        XGBoostモデルのトレーニングと評価を行うクラス
        
        Args:
            dataset_name (str): 使用するデータセット名
            random_state (int): 再現性のための乱数シード
            holdout_size (float): 独立した最終評価用データセットの割合
        """
        self.dataset_name = dataset_name
        self.random_state = random_state
        self.holdout_size = holdout_size
        self.data = None
        self.X = None
        self.y = None
        
        # 開発用と最終評価用のデータセット
        self.X_dev = None
        self.y_dev = None
        self.X_holdout = None
        self.y_holdout = None
        
        # 結果を保存するリスト
        self.acc_results = []
        self.cal_results = []
        self.auc_results = []
        self.brier_results = []
        
        # 最良のモデル情報
        self.best_model = None
        self.best_cal_error = float('inf')
        self.best_model_path = None  # 最良モデルの一時保存パス
        self.best_model_iteration = None  # 最良モデルのイテレーション
        
        # モデル保存用のディレクトリを確保
        self.models_dir = "../../Models"
        self.temp_dir = "../../Models/temp"
        os.makedirs(self.temp_dir, exist_ok=True)
        
    def load_data(self):
        """データベースからデータを読み込み、開発用と最終評価用に分割する"""
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
        
        # 開発用と最終評価用に分割（層化サンプリングで）
        self.X_dev, self.X_holdout, self.y_dev, self.y_holdout = train_test_split(
            self.X, self.y, 
            test_size=self.holdout_size, 
            random_state=self.random_state, 
            stratify=self.y
        )
        
        print(f"データセット分割:")
        print(f"  開発用データ: {self.X_dev.shape[0]}サンプル")
        print(f"  最終評価用データ: {self.X_holdout.shape[0]}サンプル（モデル開発中は使用しません）")
        
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
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=20)
        calibration_error = np.mean(np.abs(prob_true - prob_pred))
        
        # 追加の指標
        brier_score = brier_score_loss(y_true, y_pred_proba)
        log_loss_score = log_loss(y_true, y_pred_proba)
        
        return {
            'calibration_error': calibration_error,
            'brier_score': brier_score,
            'log_loss': log_loss_score
        }
    
    def plot_calibration(self, y_true, y_pred_proba, metrics, iteration, title_prefix=""):
        """
        キャリブレーションプロットを作成
        
        Args:
            y_true (array): 真のラベル
            y_pred_proba (array): 予測確率
            metrics (dict): キャリブレーション指標
            iteration (int or str): 現在のイテレーション
            title_prefix (str): タイトルの接頭辞
        """
        plt.figure(figsize=(12, 5))
        
        # キャリブレーション曲線
        plt.subplot(121)
        CalibrationDisplay.from_predictions(
            y_true, y_pred_proba, 
            n_bins=20, 
            name=f'Iteration {iteration}',
            ax=plt.gca()
        )
        plt.title(f'{title_prefix}Calibration Curve\nCal Error: {metrics["calibration_error"]:.4f}')
        
        # 予測確率ヒストグラム
        plt.subplot(122)
        plt.hist([y_pred_proba[y_true == 0], y_pred_proba[y_true == 1]], 
                 label=['Negative', 'Positive'], 
                 bins=30, alpha=0.7)
        plt.title(f'{title_prefix}Predicted Probability Distribution')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.tight_layout()
        
        # ファイル名にプレフィックスを追加
        prefix = title_prefix.replace(" ", "_").lower() if title_prefix else ""
        plt.savefig(f'{prefix}calibration_plot_iter_{iteration}.png')
        plt.close()
    
    def save_temporary_model(self, model, metrics, iteration):
        """
        一時的にモデルを保存する
        
        Args:
            model: 保存するXGBoostモデル
            metrics (dict): 評価指標
            iteration (int): 現在のイテレーション
            
        Returns:
            str: 保存したモデルのパス
        """
        # 一時保存用のファイル名を作成
        temp_model_path = f'{self.temp_dir}/TEMP_XGBoost_CAL{metrics["calibration_error"]*100:.2f}%_ACC{metrics["acc"]*100:.1f}%_ML-v{iteration}.json'
        
        # 前回の一時保存モデルがあれば削除
        if self.best_model_path and os.path.exists(self.best_model_path):
            try:
                os.remove(self.best_model_path)
            except Exception as e:
                print(f"警告: 前回の一時保存モデルの削除に失敗しました: {e}")
        
        # モデルを保存
        model.save_model(temp_model_path)
        return temp_model_path
    
    def train_and_evaluate(self, num_iterations=1000):
        """
        モデルのトレーニングと評価を実行
        
        Args:
            num_iterations (int): 学習の繰り返し回数
        """
        # データ読み込みと分割
        self.load_data()
        
        print("モデル学習開始 - キャリブレーションエラーを指標に最適化...")
        print("注意: 開発用データのみを使用し、最終評価用データは使用しません")
        print(f"モデルは一時的に {self.temp_dir} に保存され、最終評価後に適切な名前でコピーされます")
        
        # タイムスタンプ生成
        timestamp = int(time.time())
        
        for iteration in tqdm(range(num_iterations)):
            random_seed = random.randint(1, 10000) + iteration
            # 開発用データから訓練データとバリデーションデータに分割
            X_train, X_val, y_train, y_val = train_test_split(
                self.X_dev, self.y_dev, 
                test_size=0.25,  # 開発データの25%をバリデーションに使用
                random_state=random_seed,
                stratify=self.y_dev
            )
            
            # XGBoostデータマトリックスの作成
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            # モデルパラメータ
            param = {
                'max_depth': 3,
                'eta': 0.01,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'alpha': 0.1,  # L1正則化
                'lambda': 0.1,  # L2正則化
                'seed': random_seed
            }
            
            # モデルトレーニング
            model = xgb.train(
                param, 
                dtrain, 
                num_boost_round=750,
                evals=[(dval, 'val')],
                early_stopping_rounds=50,
                verbose_eval=False
            )
            
            # バリデーションデータでの予測
            y_pred_proba = model.predict(dval)
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            # 評価指標の計算
            acc = accuracy_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_pred_proba)
            
            # キャリブレーション指標の計算
            cal_metrics = self.calculate_calibration_metrics(y_val, y_pred_proba)
            
            # 結果の記録
            self.acc_results.append(acc)
            self.cal_results.append(cal_metrics['calibration_error'])
            self.auc_results.append(auc)
            self.brier_results.append(cal_metrics['brier_score'])
            
            # キャリブレーションエラーが最小のモデルを保存
            if cal_metrics['calibration_error'] < self.best_cal_error:
                self.best_cal_error = cal_metrics['calibration_error']
                self.best_model = model
                self.best_model_iteration = iteration
                
                # モデルメトリクスを保存
                self.best_model_metrics = {
                    'cal_error': cal_metrics["calibration_error"],
                    'acc': acc,
                    'auc': auc,
                    'iteration': iteration
                }
                
                # 一時的にモデルを保存（クラッシュ対策）
                self.best_model_path = self.save_temporary_model(model, {
                    'calibration_error': cal_metrics["calibration_error"],
                    'acc': acc
                }, iteration)
                
                # キャリブレーションプロットの作成
                self.plot_calibration(y_val, y_pred_proba, cal_metrics, iteration, "Validation ")
                
                print(f"\n★ 新しい最良モデルを発見（一時保存: {os.path.basename(self.best_model_path)}）")
                print(f"キャリブレーションエラー: {cal_metrics['calibration_error']:.4f}")
                print(f"精度: {acc:.4f}")
                print(f"AUC: {auc:.4f}")
                
                # モデル出力の形式をテスト
                test_pred = model.predict(xgb.DMatrix(X_val[:1]))
                print(f"テスト予測の形状: {test_pred.shape}, 値: {test_pred}")
        
        # 開発データでの最終結果の表示
        self.print_summary_results()
        
        # 最終的に、独立したホールドアウトデータセットでの評価
        self.evaluate_on_holdout_data()
    
    def print_summary_results(self):
        """
        開発データでの学習結果の概要を表示
        """
        best_idx = np.argmin(self.cal_results)
        
        print("\n=== 開発データでのトレーニング結果 ===")
        print(f"最良のキャリブレーションエラー: {self.cal_results[best_idx]:.4f}")
        print(f"対応する精度: {self.acc_results[best_idx]:.4f}")
        print(f"対応するAUC: {self.auc_results[best_idx]:.4f}")
        
        print("\n平均指標:")
        print(f"平均精度: {np.mean(self.acc_results):.4f} ± {np.std(self.acc_results):.4f}")
        print(f"平均キャリブレーションエラー: {np.mean(self.cal_results):.4f} ± {np.std(self.cal_results):.4f}")
        print(f"平均AUC: {np.mean(self.auc_results):.4f} ± {np.std(self.auc_results):.4f}")
        print(f"平均ブライアースコア: {np.mean(self.brier_results):.4f} ± {np.std(self.brier_results):.4f}")
    
    def evaluate_on_holdout_data(self):
        """
        最良のモデルを独立した最終評価用データセットで評価
        """
        print("\n=== 独立した最終評価用データセットでの評価 ===")
        print("注意: このデータはモデル開発中に一切使用されていません")
        
        if self.best_model is None or not hasattr(self, 'best_model_metrics'):
            print("最良のモデルが見つかりませんでした。最初にtrain_and_evaluate()を実行してください。")
            return
        
        # 最終評価用データでの予測
        dholdout = xgb.DMatrix(self.X_holdout, label=self.y_holdout)
        holdout_pred_proba = self.best_model.predict(dholdout)
        holdout_pred = (holdout_pred_proba >= 0.5).astype(int)
        
        # 評価指標の計算
        holdout_acc = accuracy_score(self.y_holdout, holdout_pred)
        holdout_auc = roc_auc_score(self.y_holdout, holdout_pred_proba)
        
        # キャリブレーション指標の計算
        holdout_cal_metrics = self.calculate_calibration_metrics(self.y_holdout, holdout_pred_proba)
        
        # 追加の評価指標
        holdout_precision = precision_score(self.y_holdout, holdout_pred)
        holdout_recall = recall_score(self.y_holdout, holdout_pred)
        holdout_f1 = f1_score(self.y_holdout, holdout_pred)
        
        # キャリブレーションプロットの作成
        self.plot_calibration(
            self.y_holdout, 
            holdout_pred_proba, 
            holdout_cal_metrics, 
            iteration="final", 
            title_prefix="Holdout "
        )
        
        # 結果の表示
        print(f"最終評価用データでの精度: {holdout_acc:.4f}")
        print(f"最終評価用データでのAUC: {holdout_auc:.4f}")
        print(f"最終評価用データでのキャリブレーションエラー: {holdout_cal_metrics['calibration_error']:.4f}")
        print(f"最終評価用データでのブライアースコア: {holdout_cal_metrics['brier_score']:.4f}")
        print(f"最終評価用データでの適合率(Precision): {holdout_precision:.4f}")
        print(f"最終評価用データでの再現率(Recall): {holdout_recall:.4f}")
        print(f"最終評価用データでのF1スコア: {holdout_f1:.4f}")
        
        # 開発時の指標と最終評価時の指標の比較
        dev_cal_error = self.best_model_metrics['cal_error']
        dev_acc = self.best_model_metrics['acc']
        dev_auc = self.best_model_metrics['auc']
        iteration = self.best_model_metrics['iteration']
        
        print("\n=== 開発時と最終評価時の指標比較 ===")
        print(f"精度: {dev_acc:.4f} (開発) vs {holdout_acc:.4f} (最終)")
        print(f"AUC: {dev_auc:.4f} (開発) vs {holdout_auc:.4f} (最終)")
        print(f"キャリブレーションエラー: {dev_cal_error:.4f} (開発) vs {holdout_cal_metrics['calibration_error']:.4f} (最終)")
        print(f"ブライアースコア: {self.brier_results[np.argmin(self.cal_results)]:.4f} (開発) vs {holdout_cal_metrics['brier_score']:.4f} (最終)")
        
        # 差異の計算
        acc_diff = abs(dev_acc - holdout_acc)
        auc_diff = abs(dev_auc - holdout_auc)
        cal_diff = abs(dev_cal_error - holdout_cal_metrics['calibration_error'])
        brier_diff = abs(self.brier_results[np.argmin(self.cal_results)] - holdout_cal_metrics['brier_score'])
        
        print(f"\n指標の差異(絶対値):")
        print(f"精度の差: {acc_diff:.4f}")
        print(f"AUCの差: {auc_diff:.4f}")
        print(f"キャリブレーションエラーの差: {cal_diff:.4f}")
        print(f"ブライアースコアの差: {brier_diff:.4f}")
        
        # 一般化性能のステータス判定
        generalization_status = "GOOD"
        warning_messages = []
        
        if acc_diff > 0.03:
            generalization_status = "WARNING"
            warning_messages.append(f"精度の差が大きい ({acc_diff:.4f})")
        
        if auc_diff > 0.03:
            generalization_status = "WARNING"
            warning_messages.append(f"AUCの差が大きい ({auc_diff:.4f})")
        
        if cal_diff > 0.01:
            generalization_status = "WARNING_CAL"
            warning_messages.append(f"キャリブレーションエラーの差が大きい ({cal_diff:.4f})")
        
        # 最終的なモデルを保存（ステータスをファイル名に含める）
        final_model_path = f'{self.models_dir}/{generalization_status}_2012v2_XGBoost_CAL{dev_cal_error*100:.2f}%_HOLDOUT-CAL{holdout_cal_metrics["calibration_error"]*100:.2f}%_ACC{dev_acc*100:.1f}%_HOLDOUT-ACC{holdout_acc*100:.1f}%_ML-v{iteration}.json'
        
        # 一時ファイルがあれば、最終的な名前でコピーする
        if self.best_model_path and os.path.exists(self.best_model_path):
            import shutil
            try:
                shutil.copy2(self.best_model_path, final_model_path)
                print(f"\n★ 一時保存モデルを最終モデルとして保存しました: {os.path.basename(final_model_path)}")
                
                # 一時ファイルを削除
                os.remove(self.best_model_path)
                print(f"  一時ファイル {os.path.basename(self.best_model_path)} を削除しました")
            except Exception as e:
                print(f"警告: 最終モデルの保存に失敗しました: {e}")
                # 失敗した場合は直接保存
                self.best_model.save_model(final_model_path)
                print(f"\n★ モデルを直接保存しました: {os.path.basename(final_model_path)}")
        else:
            # 一時ファイルがない場合は直接保存
            self.best_model.save_model(final_model_path)
            print(f"\n★ モデルを保存しました: {os.path.basename(final_model_path)}")
        
        # モデルの一般化性能の評価
        if generalization_status == "WARNING":
            print("\n⚠️ 注意: 開発時と最終評価時の指標に大きな差異があります。")
            for msg in warning_messages:
                print(f"- {msg}")
            print("これはモデルが開発データに過適合している可能性を示唆しています。")
            print("ハイパーパラメータの調整や正則化の強化を検討してください。")
        else:
            print("\n✓ モデルは開発データと最終評価データで一貫した性能を示しています。")
            print("これは良好な一般化能力を示唆します。")
            
    def load_best_temp_model(self):
        """
        最良の一時保存モデルを読み込む（学習が中断された場合のリカバリー用）
        
        Returns:
            bool: モデルを読み込めたかどうか
        """
        # 一時保存ディレクトリ内のファイルを列挙
        temp_files = [f for f in os.listdir(self.temp_dir) if f.startswith('TEMP_XGBoost_CAL')]
        
        if not temp_files:
            print("一時保存されたモデルが見つかりませんでした。")
            return False
        
        # キャリブレーションエラーでソート
        temp_files.sort(key=lambda x: float(x.split('CAL')[1].split('%')[0]))
        best_temp_file = os.path.join(self.temp_dir, temp_files[0])
        
        try:
            # モデルを読み込む
            self.best_model = xgb.Booster()
            self.best_model.load_model(best_temp_file)
            
            # メトリクスを抽出
            cal_error = float(best_temp_file.split('CAL')[1].split('%')[0]) / 100
            acc = float(best_temp_file.split('ACC')[1].split('%')[0]) / 100
            iteration = int(best_temp_file.split('ML-v')[1].split('.json')[0])
            
            self.best_cal_error = cal_error
            self.best_model_path = best_temp_file
            self.best_model_metrics = {
                'cal_error': cal_error,
                'acc': acc,
                'auc': 0.0,  # 正確な値は不明
                'iteration': iteration
            }
            
            print(f"一時保存モデルを読み込みました: {os.path.basename(best_temp_file)}")
            print(f"キャリブレーションエラー: {cal_error:.4f}")
            print(f"精度: {acc:.4f}")
            
            return True
        except Exception as e:
            print(f"一時保存モデルの読み込みに失敗しました: {e}")
            return False

def main():
    # ホールドアウトサイズを20%に設定
    trainer = XGBoostModelTrainer(holdout_size=0.2)
    
    # 既存の一時保存モデルがあるか確認（オプション）
    # if trainer.load_best_temp_model():
    #     # 一時保存モデルがある場合、直接評価に進むことも可能
    #     trainer.load_data()  # データは読み込む必要あり
    #     trainer.evaluate_on_holdout_data()
    # else:
    #     # 通常の学習と評価
    #     trainer.train_and_evaluate(num_iterations=1000)
    
    # イテレーション数を減らして高速にテスト（実際の使用では増やす）
    # 実際の使用では1000000など大きな値も可能
    # for _ in range(1000):
    trainer.train_and_evaluate(num_iterations=50)

if __name__ == "__main__":
    main()