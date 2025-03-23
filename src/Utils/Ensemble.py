import os
import numpy as np
import pandas as pd
import xgboost as xgb
from colorama import Fore, Style, init, deinit
from src.Utils import Expected_Value
from src.Utils import Kelly_Criterion as kc

# Initialize colorama
init()

def load_ensemble_models(models_dir="Models", prefix="XGBoost_CAL"):
    """
    指定ディレクトリから特定のプレフィックスを持つすべてのXGBoostモデルをロードする
    
    Args:
        models_dir (str): モデルファイルが格納されているディレクトリパス
        prefix (str): ロードするモデルファイルのプレフィックス
        
    Returns:
        list: ロードされたモデルのリスト
    """
    models = []
    model_files = []
    
    # ディレクトリ内のファイルを検索
    for file in os.listdir(models_dir):
        if file.startswith(prefix) and file.endswith(".json"):
            model_path = os.path.join(models_dir, file)
            model_files.append(model_path)
    
    print(f"{len(model_files)}個のモデルファイルが見つかりました")
    
    # モデルをロード
    for model_path in model_files:
        try:
            model = xgb.Booster()
            model.load_model(model_path)
            models.append({
                'model': model,
                'path': model_path,
                'cal_error': float(model_path.split('CAL')[1].split('%')[0])
            })
            print(f"モデルをロードしました: {os.path.basename(model_path)}")
        except Exception as e:
            print(f"モデルのロード中にエラーが発生しました {model_path}: {e}")
    
    return models

def ensemble_predict(models, data):
    """
    複数のモデルを使用してアンサンブル予測を行う
    
    Args:
        models (list): モデルのリスト
        data (numpy.ndarray): 予測するデータ
        
    Returns:
        dict: 予測結果と統計情報を含む辞書
    """
    all_predictions = []
    
    # 各モデルで予測を実行
    for model_info in models:
        model = model_info['model']
        try:
            prediction = model.predict(xgb.DMatrix(np.array([data])))
            
            # 予測形式の標準化
            if len(prediction.shape) > 1 and prediction.shape[1] == 2:
                # 2クラスの確率 [[p0, p1]]
                home_win_prob = prediction[0][1]
            else:
                # スカラー値の場合、ホームチーム勝利確率として解釈
                home_win_prob = prediction[0] if prediction[0] > 0.5 else 1 - prediction[0]
            
            all_predictions.append(home_win_prob)
        except Exception as e:
            print(f"予測中にエラーが発生しました: {e}")
    
    if not all_predictions:
        return None
    
    # 予測の統計情報を計算
    mean_prediction = np.mean(all_predictions)
    median_prediction = np.median(all_predictions)
    std_prediction = np.std(all_predictions)
    min_prediction = np.min(all_predictions)
    max_prediction = np.max(all_predictions)
    
    # 結果をまとめる
    return {
        'mean': mean_prediction,
        'median': median_prediction,
        'std': std_prediction,
        'min': min_prediction,
        'max': max_prediction,
        'all_predictions': all_predictions
    }

def calculate_ensemble_expected_value(ensemble_prediction, odds, adjustment_factor=0.2):
    """
    アンサンブル予測から期待値を計算する
    
    Args:
        ensemble_prediction (dict): アンサンブル予測の結果
        odds (int): アメリカンオッズ
        adjustment_factor (float): キャリブレーション調整の強さ (0.0 - 1.0)
        
    Returns:
        float: 計算された期待値
    """
    # 基本の予測確率
    base_prob = ensemble_prediction['mean']
    
    # 調整係数を小さくして、より元のモデルに近い予測に
    uncertainty_adjustment = ensemble_prediction['std'] * 1.5
    
    # 最終的な調整後の確率
    adjusted_prob = max(0.01, min(0.99, base_prob - (uncertainty_adjustment * adjustment_factor)))
    
    # 期待値を計算
    ev = Expected_Value.expected_value(adjusted_prob, odds)
    
    return ev, adjusted_prob

def ensemble_xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion):
    """
    アンサンブルXGBoostモデルを使用して試合の予測を行い、結果を表示する
    
    Args:
        data: 予測に使用するデータ
        todays_games_uo: 試合ごとのオーバー/アンダーライン
        frame_ml: モデル用のデータフレーム
        games: 試合情報のリスト
        home_team_odds: ホームチームのオッズリスト
        away_team_odds: アウェイチームのオッズリスト
        kelly_criterion: ケリー基準を使用するかどうかのフラグ
    """
    # モデルをロード
    models = load_ensemble_models(models_dir="Models", prefix="XGBoost_CAL")
    
    if not models:
        print("モデルがロードできませんでした。プログラムを終了します。")
        return
    
    print(f"アンサンブル予測に{len(models)}個のモデルを使用します")
    
    # 予測結果を格納するリスト
    predictions_list = []
    
    # オーバー/アンダーの予測用モデル
    ou_models = load_ensemble_models(models_dir="Models/XGBoost_Models", prefix="XGBoost")
    ou_model = ou_models[0]['model'] if ou_models else models[0]['model']
    
    # 各試合に対して予測
    count = 0
    for game in games:
        if count >= len(games) or count >= len(data):
            break
            
        home_team = game[0]
        away_team = game[1]
        
        # ホームチームとアウェイチームの予測
        home_prediction = ensemble_predict(models, data[count])
        
        if not home_prediction:
            print(f"予測できませんでした: {home_team} vs {away_team}")
            count += 1
            continue
        
        # オーバー/アンダー予測
        try:
            ou_pred_raw = ou_model.predict(xgb.DMatrix(np.array([data[count]])))
            
            # 予測形式の標準化
            if len(ou_pred_raw.shape) > 1 and ou_pred_raw.shape[1] == 2:
                under_prob = ou_pred_raw[0][0]
                over_prob = ou_pred_raw[0][1]
            else:
                under_prob = ou_pred_raw[0]
                over_prob = 1 - under_prob
                
            under_over = 1 if over_prob > under_prob else 0
            ou_confidence = max(under_prob, over_prob) * 100
        except Exception as e:
            print(f"オーバー/アンダー予測中にエラーが発生しました: {e}")
            under_over = 0
            ou_confidence = 50.0
        
        # 主要な予測を表示形式に変換
        winner = 1 if home_prediction['mean'] > 0.5 else 0
        winner_confidence = home_prediction['mean'] * 100 if winner == 1 else (1 - home_prediction['mean']) * 100
        
        # 予測結果の表示（XGBoost_Runnerと同じ形式）
        if winner == 1:
            if under_over == 0:
                print(
                    Fore.GREEN + home_team + Style.RESET_ALL + 
                    Fore.CYAN + f" ({winner_confidence:.1f}%)" + Style.RESET_ALL + 
                    ' vs ' + Fore.RED + away_team + Style.RESET_ALL + ': ' +
                    Fore.MAGENTA + 'UNDER ' + Style.RESET_ALL + 
                    str(todays_games_uo[count]) + Style.RESET_ALL + 
                    Fore.CYAN + f" ({ou_confidence:.1f}%)" + Style.RESET_ALL
                )
            else:
                print(
                    Fore.GREEN + home_team + Style.RESET_ALL + 
                    Fore.CYAN + f" ({winner_confidence:.1f}%)" + Style.RESET_ALL + 
                    ' vs ' + Fore.RED + away_team + Style.RESET_ALL + ': ' +
                    Fore.BLUE + 'OVER ' + Style.RESET_ALL + 
                    str(todays_games_uo[count]) + Style.RESET_ALL + 
                    Fore.CYAN + f" ({ou_confidence:.1f}%)" + Style.RESET_ALL
                )
        else:
            if under_over == 0:
                print(
                    Fore.RED + home_team + Style.RESET_ALL + 
                    ' vs ' + Fore.GREEN + away_team + Style.RESET_ALL + 
                    Fore.CYAN + f" ({winner_confidence:.1f}%)" + Style.RESET_ALL + ': ' +
                    Fore.MAGENTA + 'UNDER ' + Style.RESET_ALL + 
                    str(todays_games_uo[count]) + Style.RESET_ALL + 
                    Fore.CYAN + f" ({ou_confidence:.1f}%)" + Style.RESET_ALL
                )
            else:
                print(
                    Fore.RED + home_team + Style.RESET_ALL + 
                    ' vs ' + Fore.GREEN + away_team + Style.RESET_ALL + 
                    Fore.CYAN + f" ({winner_confidence:.1f}%)" + Style.RESET_ALL + ': ' +
                    Fore.BLUE + 'OVER ' + Style.RESET_ALL + 
                    str(todays_games_uo[count]) + Style.RESET_ALL + 
                    Fore.CYAN + f" ({ou_confidence:.1f}%)" + Style.RESET_ALL
                )
        
        # 期待値計算
        if count < len(home_team_odds) and count < len(away_team_odds) and home_team_odds[count] and away_team_odds[count]:
            # ホームチームの期待値
            home_ev, adjusted_home_prob = calculate_ensemble_expected_value(
                home_prediction, 
                home_team_odds[count]
            )
            
            # アウェイチームの期待値
            away_prediction = {
                'mean': 1 - home_prediction['mean'],
                'median': 1 - home_prediction['median'],
                'std': home_prediction['std'],
                'min': 1 - home_prediction['max'],
                'max': 1 - home_prediction['min']
            }
            
            away_ev, adjusted_away_prob = calculate_ensemble_expected_value(
                away_prediction,
                away_team_odds[count]
            )
            
            # ケリー基準
            kelly_home = kc.calculate_kelly_criterion(home_team_odds[count], adjusted_home_prob)
            kelly_away = kc.calculate_kelly_criterion(away_team_odds[count], adjusted_away_prob)
            
            # 予測データをリストに追加
            predictions_list.append({
                'home_team': home_team,
                'away_team': away_team,
                'home_win_prob': home_prediction['mean'],
                'away_win_prob': away_prediction['mean'],
                'home_ev': home_ev,
                'away_ev': away_ev,
                'kelly_home': kelly_home,
                'kelly_away': kelly_away,
                'home_odds': home_team_odds[count],
                'away_odds': away_team_odds[count]
            })
        
        count += 1
    
    # 期待値とケリー基準の表示
    if kelly_criterion:
        print("------------Expected Value & Kelly Criterion-----------")
    else:
        print("---------------------Expected Value--------------------")
        
    for pred in predictions_list:
        home_team = pred['home_team']
        away_team = pred['away_team']
        
        # 表示色の決定
        home_color = Fore.GREEN if pred['home_ev'] > 0 else Fore.RED
        away_color = Fore.GREEN if pred['away_ev'] > 0 else Fore.RED
        
        # ケリー基準の表示
        bankroll_descriptor = ' Fraction of Bankroll: '
        bankroll_fraction_home = bankroll_descriptor + str(pred['kelly_home']) + '%'
        bankroll_fraction_away = bankroll_descriptor + str(pred['kelly_away']) + '%'
        
        # 結果の表示
        print(home_team + ' EV: ' + home_color + str(pred['home_ev']) + Style.RESET_ALL + 
              (bankroll_fraction_home if kelly_criterion else ''))
        print(away_team + ' EV: ' + away_color + str(pred['away_ev']) + Style.RESET_ALL + 
              (bankroll_fraction_away if kelly_criterion else ''))
    
    # 推奨ベットのサマリー
    print("------- 推奨ベット（勝率50%以上かつプラスのEV）-------")
    recommended_bets = []
    
    for pred in predictions_list:
        # 推奨ベットの条件を修正: EVがプラスならば勝率に関わらず推奨
        if pred['home_ev'] > 0:
            recommended_bets.append({
                'team': pred['home_team'],
                'opponent': pred['away_team'],
                'win_prob': pred['home_win_prob'] * 100,
                'ev': pred['home_ev'],
                'kelly': pred['kelly_home'],
                'odds': pred['home_odds']
            })
        if pred['away_ev'] > 0:
            recommended_bets.append({
                'team': pred['away_team'],
                'opponent': pred['home_team'],
                'win_prob': pred['away_win_prob'] * 100,
                'ev': pred['away_ev'],
                'kelly': pred['kelly_away'],
                'odds': pred['away_odds']
            })
    
    # EVの高い順にソート
    recommended_bets.sort(key=lambda x: x['ev'], reverse=True)
    
    if recommended_bets:
        for bet in recommended_bets:
            print(f"{bet['team']} vs {bet['opponent']}: " + 
                  Fore.CYAN + f"勝率 {bet['win_prob']:.1f}%" + Style.RESET_ALL + ", " + 
                  Fore.GREEN + f"EV {bet['ev']:.2f}" + Style.RESET_ALL + ", " + 
                  f"ケリー {bet['kelly']:.2f}%, オッズ {bet['odds']}")
    else:
        print("条件を満たすベットはありません")
    
    print("------------------------------------------------------")
    
    deinit()