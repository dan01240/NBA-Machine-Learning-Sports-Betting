import copy
import numpy as np
import pandas as pd
import xgboost as xgb
from colorama import Fore, Style, Back, init, deinit
import csv
import os
from datetime import datetime
from src.Utils import Expected_Value
from src.Utils import Kelly_Criterion as kc

# Initialize colorama
init()

# Load XGBoost models
xgb_ml = xgb.Booster()
# xgb_ml.load_model('Models/ACC/A_XGBoost_CAL6.84%_ACC72.3%_ML-v573.json')
xgb_ml.load_model('Models/GOOD_2012v2_XGBoost_CAL1.83%_HOLDOUT-CAL2.75%_ACC64.3%_HOLDOUT-ACC64.1%_ML-v28.json')
xgb_uo = xgb.Booster()
xgb_uo.load_model('Models/XGBoost_Models/XGBoost_53.7%_UO-9.json')

def xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion):
    """
    XGBoostモデルを使用して試合の予測を行い、結果を表示する関数
    条件を満たすベット（勝率50%以上かつEVがプラス）を強調表示
    
    Args:
        data: 予測に使用するデータ
        todays_games_uo: 試合ごとのオーバー/アンダーライン
        frame_ml: モデル用のデータフレーム
        games: 試合情報のリスト
        home_team_odds: ホームチームのオッズリスト
        away_team_odds: アウェイチームのオッズリスト
        kelly_criterion: ケリー基準を使用するかどうかのフラグ
    """
    ml_predictions_array = []
    
    # 勝敗予測の実行
    for row in data:
        ml_predictions_array.append(xgb_ml.predict(xgb.DMatrix(np.array([row]))))
    
    frame_uo = copy.deepcopy(frame_ml)
    frame_uo['OU'] = np.asarray(todays_games_uo)
    data = frame_uo.values
    data = data.astype(float)

    ou_predictions_array = []

    for row in data:
        ou_predictions_array.append(xgb_uo.predict(xgb.DMatrix(np.array([row]))))

    # 予測結果のリストを初期化
    predictions_list = []
    
    count = 0
    for game in games:
        home_team = game[0]
        away_team = game[1]
        
        # 勝者予測
        winner = int(np.argmax(ml_predictions_array[count]))
        under_over = int(np.argmax(ou_predictions_array[count]))
        
        # 予測確率の取得
        winner_confidence = ml_predictions_array[count]
        un_confidence = ou_predictions_array[count]
        
        # 期待値とケリー基準を計算
        ev_home = ev_away = 0
        kelly_home = kelly_away = 0
        if home_team_odds[count] and away_team_odds[count]:
            ev_home = float(Expected_Value.expected_value(ml_predictions_array[count][0][1], int(home_team_odds[count])))
            ev_away = float(Expected_Value.expected_value(ml_predictions_array[count][0][0], int(away_team_odds[count])))
            kelly_home = kc.calculate_kelly_criterion(home_team_odds[count], ml_predictions_array[count][0][1])
            kelly_away = kc.calculate_kelly_criterion(away_team_odds[count], ml_predictions_array[count][0][0])
        
        # 各チームについて勝率と期待値の条件を確認
        home_win_prob = winner_confidence[0][1] * 100 if len(winner_confidence[0]) > 1 else (1 - winner_confidence[0][0]) * 100
        away_win_prob = winner_confidence[0][0] * 100 if len(winner_confidence[0]) > 1 else winner_confidence[0][0] * 100
        
        home_meets_criteria = home_win_prob > 50 and ev_home > 0
        away_meets_criteria = away_win_prob > 50 and ev_away > 0
        
        # 予測データを辞書に保存
        predictions_list.append({
            'home_team': home_team,
            'away_team': away_team,
            'winner': 'home' if winner == 1 else 'away',
            'home_win_prob': home_win_prob,
            'away_win_prob': away_win_prob,
            'over_under': 'over' if under_over == 1 else 'under',
            'over_under_line': todays_games_uo[count],
            'over_under_prob': un_confidence[0][1] * 100 if under_over == 1 else un_confidence[0][0] * 100,
            'home_odds': home_team_odds[count],
            'away_odds': away_team_odds[count],
            'ev_home': ev_home,
            'ev_away': ev_away,
            'kelly_home': kelly_home,
            'kelly_away': kelly_away,
            'home_meets_criteria': home_meets_criteria,
            'away_meets_criteria': away_meets_criteria
        })
        
        # 予測結果の表示（ANSI色付き）
        if winner == 1:
            # ホームチーム勝利予測
            winner_confidence_val = round(winner_confidence[0][1] * 100, 1)
            
            # ホームチームが条件を満たす場合は背景色を変更
            home_format = Back.GREEN + Fore.WHITE if home_meets_criteria else Fore.GREEN
            
            if under_over == 0:
                # アンダー予測
                un_confidence_val = round(ou_predictions_array[count][0][0] * 100, 1)
                print(
                    home_format + home_team + Style.RESET_ALL + 
                    Fore.CYAN + f" ({winner_confidence_val}%)" + Style.RESET_ALL + 
                    ' vs ' + Fore.RED + away_team + Style.RESET_ALL + ': ' +
                    Fore.MAGENTA + 'UNDER ' + Style.RESET_ALL + 
                    str(todays_games_uo[count]) + Style.RESET_ALL + 
                    Fore.CYAN + f" ({un_confidence_val}%)" + Style.RESET_ALL
                )
            else:
                # オーバー予測
                un_confidence_val = round(ou_predictions_array[count][0][1] * 100, 1)
                print(
                    home_format + home_team + Style.RESET_ALL + 
                    Fore.CYAN + f" ({winner_confidence_val}%)" + Style.RESET_ALL + 
                    ' vs ' + Fore.RED + away_team + Style.RESET_ALL + ': ' +
                    Fore.BLUE + 'OVER ' + Style.RESET_ALL + 
                    str(todays_games_uo[count]) + Style.RESET_ALL + 
                    Fore.CYAN + f" ({un_confidence_val}%)" + Style.RESET_ALL
                )
        else:
            # アウェイチーム勝利予測
            winner_confidence_val = round(winner_confidence[0][0] * 100, 1)
            
            # アウェイチームが条件を満たす場合は背景色を変更
            away_format = Back.GREEN + Fore.WHITE if away_meets_criteria else Fore.GREEN
            
            if under_over == 0:
                # アンダー予測
                un_confidence_val = round(ou_predictions_array[count][0][0] * 100, 1)
                print(
                    Fore.RED + home_team + Style.RESET_ALL + 
                    ' vs ' + away_format + away_team + Style.RESET_ALL + 
                    Fore.CYAN + f" ({winner_confidence_val}%)" + Style.RESET_ALL + ': ' +
                    Fore.MAGENTA + 'UNDER ' + Style.RESET_ALL + 
                    str(todays_games_uo[count]) + Style.RESET_ALL + 
                    Fore.CYAN + f" ({un_confidence_val}%)" + Style.RESET_ALL
                )
            else:
                # オーバー予測
                un_confidence_val = round(ou_predictions_array[count][0][1] * 100, 1)
                print(
                    Fore.RED + home_team + Style.RESET_ALL + 
                    ' vs ' + away_format + away_team + Style.RESET_ALL + 
                    Fore.CYAN + f" ({winner_confidence_val}%)" + Style.RESET_ALL + ': ' +
                    Fore.BLUE + 'OVER ' + Style.RESET_ALL + 
                    str(todays_games_uo[count]) + Style.RESET_ALL + 
                    Fore.CYAN + f" ({un_confidence_val}%)" + Style.RESET_ALL
                )
        count += 1

    # 期待値とケリー基準の表示
    if kelly_criterion:
        print("------------Expected Value & Kelly Criterion-----------")
    else:
        print("---------------------Expected Value--------------------")
    
    count = 0
    for game in games:
        home_team = game[0]
        away_team = game[1]
        
        pred = predictions_list[count]
        ev_home = pred['ev_home']
        ev_away = pred['ev_away']
        kelly_home = pred['kelly_home']
        kelly_away = pred['kelly_away']
        
        # 表示色の決定
        expected_value_colors = {
            'home_color': Fore.GREEN if ev_home > 0 else Fore.RED,
            'away_color': Fore.GREEN if ev_away > 0 else Fore.RED
        }
        
        # 条件を満たすチームは背景色を変更
        home_format = Back.GREEN + Fore.WHITE if pred['home_meets_criteria'] else expected_value_colors['home_color']
        away_format = Back.GREEN + Fore.WHITE if pred['away_meets_criteria'] else expected_value_colors['away_color']
        
        # 結果の表示
        bankroll_descriptor = ' Fraction of Bankroll: '
        bankroll_fraction_home = bankroll_descriptor + str(kelly_home) + '%'
        bankroll_fraction_away = bankroll_descriptor + str(kelly_away) + '%'

        print(home_team + ' EV: ' + home_format + str(ev_home) + Style.RESET_ALL + (bankroll_fraction_home if kelly_criterion else ''))
        print(away_team + ' EV: ' + away_format + str(ev_away) + Style.RESET_ALL + (bankroll_fraction_away if kelly_criterion else ''))
        count += 1
    
    # 条件を満たすベットのみを表示
    print("\n------- 推奨ベット（勝率50%以上かつプラスのEV）-------")
    recommended_bets = []
    for pred in predictions_list:
        if pred['home_meets_criteria']:
            recommended_bets.append({
                'team': pred['home_team'],
                'opponent': pred['away_team'],
                'win_prob': pred['home_win_prob'],
                'ev': pred['ev_home'],
                'kelly': pred['kelly_home'],
                'odds': pred['home_odds']
            })
        if pred['away_meets_criteria']:
            recommended_bets.append({
                'team': pred['away_team'],
                'opponent': pred['home_team'],
                'win_prob': pred['away_win_prob'],
                'ev': pred['ev_away'],
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
    
    # CSV出力
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"predictions_{timestamp}.csv"
    
    # 出力ディレクトリがなければ作成
    output_dir = "predictions"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # ヘッダー行を書き込み
        writer.writerow([
            'Date', 'Home Team', 'Away Team', 
            'Predicted Winner', 'Win Confidence', 
            'Total Line', 'O/U Prediction', 'O/U Confidence',
            'Home Odds', 'Away Odds', 
            'Home EV', 'Away EV',
            'Home Kelly %', 'Away Kelly %',
            'Recommended Bet'
        ])
        
        # 予測データの書き込み
        for i, pred in enumerate(predictions_list):
            # 推奨ベットを判断
            home_recommended = pred['home_meets_criteria']
            away_recommended = pred['away_meets_criteria']
            recommended = ""
            
            if home_recommended and away_recommended:
                if pred['ev_home'] > pred['ev_away']:
                    recommended = f"{pred['home_team']} (EV: {pred['ev_home']:.2f})"
                else:
                    recommended = f"{pred['away_team']} (EV: {pred['ev_away']:.2f})"
            elif home_recommended:
                recommended = f"{pred['home_team']} (EV: {pred['ev_home']:.2f})"
            elif away_recommended:
                recommended = f"{pred['away_team']} (EV: {pred['ev_away']:.2f})"
            
            # 試合の日付（デフォルトは今日）
            game_date = datetime.now().strftime("%Y-%m-%d")
            
            # 行データの書き込み
            writer.writerow([
                game_date, 
                pred['home_team'], 
                pred['away_team'],
                pred['home_team'] if pred['winner'] == 'home' else pred['away_team'], 
                f"{pred['home_win_prob']:.1f}%" if pred['winner'] == 'home' else f"{pred['away_win_prob']:.1f}%",
                pred['over_under_line'],
                pred['over_under'].upper(), 
                f"{pred['over_under_prob']:.1f}%",
                pred['home_odds'], 
                pred['away_odds'],
                f"{pred['ev_home']:.2f}", 
                f"{pred['ev_away']:.2f}",
                f"{pred['kelly_home']:.2f}%", 
                f"{pred['kelly_away']:.2f}%",
                recommended
            ])
    
    print(f"CSV出力が完了しました: {filepath}")
    deinit()