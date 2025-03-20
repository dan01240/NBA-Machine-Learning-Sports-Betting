import copy

import numpy as np
import pandas as pd
import xgboost as xgb
from colorama import Fore, Style, init, deinit
from src.Utils import Expected_Value
from src.Utils import Kelly_Criterion as kc


# from src.Utils.Dictionaries import team_index_current
# from src.Utils.tools import get_json_data, to_data_frame, get_todays_games_json, create_todays_games
init()
xgb_ml = xgb.Booster()
xgb_ml.load_model('Models/CAL/XGBoost_CAL1.31%_ACC61.7%_ML-1742435642.json')
xgb_uo = xgb.Booster()
xgb_uo.load_model('Models/XGBoost_Models/XGBoost_53.7%_UO-9.json')


def xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion):
    ml_predictions_array = []

    for row in data:
        ml_predictions_array.append(xgb_ml.predict(xgb.DMatrix(np.array([row]))))
    # 予測確率の形状を出力
    print("ML Predictions Shape:", [pred.shape for pred in ml_predictions_array])
    print("First Prediction:", ml_predictions_array[0])
    frame_uo = copy.deepcopy(frame_ml)
    frame_uo['OU'] = np.asarray(todays_games_uo)
    data = frame_uo.values
    data = data.astype(float)

    ou_predictions_array = []

    for row in data:
        ou_predictions_array.append(xgb_uo.predict(xgb.DMatrix(np.array([row]))))

    count = 0
    for game in games:
        home_team = game[0]
        away_team = game[1]
        winner = int(np.argmax(ml_predictions_array[count]))
        under_over = int(np.argmax(ou_predictions_array[count]))
        winner_confidence = ml_predictions_array[count]
        un_confidence = ou_predictions_array[count]
        if winner == 1:
            winner_confidence = round(winner_confidence[0][1] * 100, 1)
            if under_over == 0:
                un_confidence = round(ou_predictions_array[count][0][0] * 100, 1)
                print(
                    Fore.GREEN + home_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL + ' vs ' + Fore.RED + away_team + Style.RESET_ALL + ': ' +
                    Fore.MAGENTA + 'UNDER ' + Style.RESET_ALL + str(
                        todays_games_uo[count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)" + Style.RESET_ALL)
            else:
                un_confidence = round(ou_predictions_array[count][0][1] * 100, 1)
                print(
                    Fore.GREEN + home_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL + ' vs ' + Fore.RED + away_team + Style.RESET_ALL + ': ' +
                    Fore.BLUE + 'OVER ' + Style.RESET_ALL + str(
                        todays_games_uo[count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)" + Style.RESET_ALL)
        else:
            winner_confidence = round(winner_confidence[0][0] * 100, 1)
            if under_over == 0:
                un_confidence = round(ou_predictions_array[count][0][0] * 100, 1)
                print(
                    Fore.RED + home_team + Style.RESET_ALL + ' vs ' + Fore.GREEN + away_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL + ': ' +
                    Fore.MAGENTA + 'UNDER ' + Style.RESET_ALL + str(
                        todays_games_uo[count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)" + Style.RESET_ALL)
            else:
                un_confidence = round(ou_predictions_array[count][0][1] * 100, 1)
                print(
                    Fore.RED + home_team + Style.RESET_ALL + ' vs ' + Fore.GREEN + away_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL + ': ' +
                    Fore.BLUE + 'OVER ' + Style.RESET_ALL + str(
                        todays_games_uo[count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)" + Style.RESET_ALL)
        count += 1

    if kelly_criterion:
        print("------------Expected Value & Kelly Criterion-----------")
    else:
        print("---------------------Expected Value--------------------")
    count = 0
    for game in games:
        home_team = game[0]
        away_team = game[1]
        ev_home = ev_away = 0
        if home_team_odds[count] and away_team_odds[count]:
            ev_home = float(Expected_Value.expected_value(ml_predictions_array[count][0][1], int(home_team_odds[count])))
            ev_away = float(Expected_Value.expected_value(ml_predictions_array[count][0][0], int(away_team_odds[count])))
        expected_value_colors = {'home_color': Fore.GREEN if ev_home > 0 else Fore.RED,
                        'away_color': Fore.GREEN if ev_away > 0 else Fore.RED}
        bankroll_descriptor = ' Fraction of Bankroll: '
        bankroll_fraction_home = bankroll_descriptor + str(kc.calculate_kelly_criterion(home_team_odds[count], ml_predictions_array[count][0][1])) + '%'
        bankroll_fraction_away = bankroll_descriptor + str(kc.calculate_kelly_criterion(away_team_odds[count], ml_predictions_array[count][0][0])) + '%'

        print(home_team + ' EV: ' + expected_value_colors['home_color'] + str(ev_home) + Style.RESET_ALL + (bankroll_fraction_home if kelly_criterion else ''))
        print(away_team + ' EV: ' + expected_value_colors['away_color'] + str(ev_away) + Style.RESET_ALL + (bankroll_fraction_away if kelly_criterion else ''))
        # 期待値(EV):
        # 予測確率とオッズから計算される理論的な収益
        # 正の値はプラスの期待値（長期的に利益が出る）を示す
        # 例：Dallas Mavericks EV: 84.40 → $100賭けた場合の期待値は$84.40
        
        # ケリー基準(Kelly Criterion):
        # 最適な賭け金額の割合を計算
        # Fraction of Bankroll: 総資金のうち何%を賭けるべきかを示す
        # 例：Dallas Mavericks ... Fraction of Bankroll: 22.51% → 資金の22.51%を賭けるのが最適
        # Dallas Mavericks (EV: 84.40) - 最も高いEV、資金の22.51%を賭けることが推奨
        # San Antonio Spurs (EV: 109.93) - 非常に高いEV、資金の35.46%を賭けることが推奨
        # Washington Wizards (EV: 34.46) - 良好なEV、資金の22.23%を賭けることが推奨
        # Miami Heat (EV: 32.06) - 良好なEV、資金の20.68%を賭けることが推奨
        # New Orleans Pelicans (EV: 30.73) - 良好なEV、資金の5.35%を賭けることが推奨
        # Indiana Pacers EV: -29.579999923706055 Fraction of Bankroll: 0%
        # Dallas Mavericks EV: 74.9000015258789 Fraction of Bankroll: 19.97%
        # Orlando Magic EV: -0.23000000417232513 Fraction of Bankroll: 0%
        # Houston Rockets EV: -13.670000076293945 Fraction of Bankroll: 0%
        # Miami Heat EV: 26.959999084472656 Fraction of Bankroll: 17.39%
        # Detroit Pistons EV: -28.809999465942383 Fraction of Bankroll: 0%
        # Minnesota Timberwolves EV: -14.069999694824219 Fraction of Bankroll: 0%
        # New Orleans Pelicans EV: 17.229999542236328 Fraction of Bankroll: 3.0%
        # San Antonio Spurs EV: 101.7300033569336 Fraction of Bankroll: 32.82%
        # New York Knicks EV: -41.20000076293945 Fraction of Bankroll: 0%
        # Oklahoma City Thunder EV: -10.619999885559082 Fraction of Bankroll: 0%
        # Philadelphia 76ers EV: -3.490000009536743 Fraction of Bankroll: 0%
        # Utah Jazz EV: -30.260000228881836 Fraction of Bankroll: 0%
        # Washington Wizards EV: 29.360000610351562 Fraction of Bankroll: 18.94%
        # Los Angeles Lakers EV: 10.050000190734863 Fraction of Bankroll: 9.14%
        # Denver Nuggets EV: -22.8700008392334 Fraction of Bankroll: 0%
        # Sacramento Kings EV: 9.010000228881836 Fraction of Bankroll: 5.15%
        # Cleveland Cavaliers EV: -16.799999237060547 Fraction of Bankroll: 0%
        # Phoenix Suns EV: -9.170000076293945 Fraction of Bankroll: 0%
        # Chicago Bulls EV: -7.409999847412109 Fraction of Bankroll: 0%
        # Portland Trail Blazers EV: -8.020000457763672 Fraction of Bankroll: 0%
        # Memphis Grizzlies EV: -8.4399995803833 Fraction of Bankroll: 0%
        count += 1

    deinit()
