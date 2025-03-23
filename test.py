# main.py の修正部分

import argparse
from datetime import datetime, timedelta

import pandas as pd
import tensorflow as tf
from colorama import Fore, Style

from src.DataProviders.SbrOddsProvider import SbrOddsProvider
from src.Predict import NN_Runner, XGBoost_Runner
from src.Utils.Dictionaries import team_index_current
from src.Utils.tools import create_todays_games_from_odds, get_json_data, to_data_frame, get_todays_games_json, create_todays_games

# https://www.nj.bet365.com/#/AC/B18/C20604387/D48/E1453/F10
todays_games_url = 'https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/2024/scores/00_todays_scores.json'
data_url = 'https://stats.nba.com/stats/leaguedashteamstats?' \
           'Conference=&DateFrom=&DateTo=&Division=&GameScope=&' \
           'GameSegment=&LastNGames=0&LeagueID=00&Location=&' \
           'MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&' \
           'PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&' \
           'PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&' \
           'Season=2024-25&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&' \
           'StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='

# 新しく追加するテストデータ
test_games = [
    # Wed Mar 19
    # ["Utah Jazz", "Washington Wizards"],
    # ["Phoenix Suns", "Chicago Bulls"],
    # ["Sacramento Kings", "Cleveland Cavaliers"],
    # ["Los Angeles Lakers", "Denver Nuggets"],
    # ["Portland Trail Blazers", "Memphis Grizzlies"],
    # # Thu Mar 20
    ["Indiana Pacers", "Brooklyn Nets"],
    ["Charlotte Hornets", "New York Knicks"],
    ["Sacramento Kings", "Chicago Bulls"],
    ["Golden State Warriors", "Toronto Raptors"],
    ["Los Angeles Lakers", "Milwaukee Bucks"]
]

test_odds = {
    # Wed Mar 19
    # "Utah Jazz": {"money_line_odds": -6600, "spread": -11.5, "total": 243.5},
    # "Washington Wizards": {"money_line_odds": 1600, "spread": 11.5, "total": 243.5},
    # "Phoenix Suns": {"money_line_odds": -800, "spread": -12.5, "total": 230.5},
    # "Chicago Bulls": {"money_line_odds": 500, "spread": 12.5, "total": 230.5},
    # "Sacramento Kings": {"money_line_odds": 300, "spread": 6.5, "total": 221.5},
    # "Cleveland Cavaliers": {"money_line_odds": -400, "spread": -6.5, "total": 221.5},
    # "Los Angeles Lakers": {"money_line_odds": -6600, "spread": -21.5, "total": 247.5},
    # "Denver Nuggets": {"money_line_odds": 1600, "spread": 21.5, "total": 247.5},
    # "Portland Trail Blazers": {"money_line_odds": -425, "spread": -7.5, "total": 252.5},
    # "Memphis Grizzlies": {"money_line_odds": 310, "spread": 7.5, "total": 252.5},
    # # Thu Mar 20
    "Indiana Pacers": {"money_line_odds": -400, "spread": -9.0, "total": 227.0},
    "Brooklyn Nets": {"money_line_odds": 320, "spread": 9.0, "total": 227.0},
    "Charlotte Hornets": {"money_line_odds": 240, "spread": 7.5, "total": 222.5},
    "New York Knicks": {"money_line_odds": -300, "spread": -7.5, "total": 222.5},
    "Chicago Bulls (Thu)": {"money_line_odds": 210, "spread": 6.5, "total": 235.5},
    "Sacramento Kings (Thu)": {"money_line_odds": -260, "spread": -6.5, "total": 235.5},
    "Golden State Warriors": {"money_line_odds": -900, "spread": -14.0, "total": 226.5},
    "Toronto Raptors": {"money_line_odds": 600, "spread": 14.0, "total": 226.5},
    "Milwaukee Bucks": {"money_line_odds": -150, "spread": -3.0, "total": 226.5},
    "Los Angeles Lakers (Thu)": {"money_line_odds": 125, "spread": 3.0, "total": 226.5}
}


def createTodaysGames(games, df, odds, use_test_data=False):
    match_data = []
    todays_games_uo = []
    home_team_odds = []
    away_team_odds = []

    home_team_days_rest = []
    away_team_days_rest = []

    for game in games:
        home_team = game[0]
        away_team = game[1]
        if home_team not in team_index_current or away_team not in team_index_current:
            continue
            
        # テストデータを使用する場合
        if use_test_data:
            # 試合の総得点(オーバー/アンダーライン)を設定
            home_team_total = test_odds.get(home_team, {}).get("total")
            away_team_total = test_odds.get(away_team, {}).get("total")
            # どちらかがあればそれを使用
            total = home_team_total or away_team_total or 225.5
            todays_games_uo.append(total)
            
            # モネーラインオッズを設定
            home_team_odds.append(test_odds.get(home_team, {}).get("money_line_odds", -110))
            away_team_odds.append(test_odds.get(away_team, {}).get("money_line_odds", -110))
        # APIから取得したオッズデータを使用する場合
        elif odds is not None:
            game_odds = odds.get(home_team + ':' + away_team, {})
            todays_games_uo.append(game_odds.get('under_over_odds'))
            home_team_odds.append(game_odds.get(home_team, {}).get('money_line_odds'))
            away_team_odds.append(game_odds.get(away_team, {}).get('money_line_odds'))
        # オッズデータがない場合は手動入力
        else:
            todays_games_uo.append(input(home_team + ' vs ' + away_team + ': '))
            home_team_odds.append(input(home_team + ' odds: '))
            away_team_odds.append(input(away_team + ' odds: '))

        # 休日数の計算
        try:
            schedule_df = pd.read_csv('Data/nba-2024-UTC.csv', parse_dates=['Date'], date_format='%d/%m/%Y %H:%M')
            home_games = schedule_df[(schedule_df['Home Team'] == home_team) | (schedule_df['Away Team'] == home_team)]
            away_games = schedule_df[(schedule_df['Home Team'] == away_team) | (schedule_df['Away Team'] == away_team)]
            previous_home_games = home_games.loc[schedule_df['Date'] <= datetime.today()].sort_values('Date',ascending=False).head(1)['Date']
            previous_away_games = away_games.loc[schedule_df['Date'] <= datetime.today()].sort_values('Date',ascending=False).head(1)['Date']
            
            if len(previous_home_games) > 0:
                last_home_date = previous_home_games.iloc[0]
                home_days_off = timedelta(days=1) + datetime.today() - last_home_date
            else:
                home_days_off = timedelta(days=2)  # デフォルト値
                
            if len(previous_away_games) > 0:
                last_away_date = previous_away_games.iloc[0]
                away_days_off = timedelta(days=1) + datetime.today() - last_away_date
            else:
                away_days_off = timedelta(days=2)  # デフォルト値
                
            # print(f"{away_team} days off: {away_days_off.days} @ {home_team} days off: {home_days_off.days}")
        except Exception as e:
            print(f"Error calculating rest days: {e}")
            home_days_off = timedelta(days=2)  # デフォルト値
            away_days_off = timedelta(days=2)  # デフォルト値

        home_team_days_rest.append(home_days_off.days)
        away_team_days_rest.append(away_days_off.days)
        home_team_series = df.iloc[team_index_current.get(home_team)]
        away_team_series = df.iloc[team_index_current.get(away_team)]
        stats = pd.concat([home_team_series, away_team_series])
        stats['Days-Rest-Home'] = home_days_off.days
        stats['Days-Rest-Away'] = away_days_off.days
        match_data.append(stats)

    games_data_frame = pd.concat(match_data, ignore_index=True, axis=1)
    games_data_frame = games_data_frame.T

    frame_ml = games_data_frame.drop(columns=['TEAM_ID', 'TEAM_NAME'])
    data = frame_ml.values
    data = data.astype(float)

    return data, todays_games_uo, frame_ml, home_team_odds, away_team_odds


def main():
    odds = None
    
    # テストデータを使用するかどうか
    if args.test:
        print(Fore.CYAN + "テストデータを使用します" + Style.RESET_ALL)
        games = test_games
        use_test_data = True
    else:
        use_test_data = False
        if args.odds:
            odds = SbrOddsProvider(sportsbook=args.odds).get_odds()
            games = create_todays_games_from_odds(odds)
            if len(games) == 0:
                print("No games found.")
                return
            if (games[0][0] + ':' + games[0][1]) not in list(odds.keys()):
                print(games[0][0] + ':' + games[0][1])
                print(Fore.RED,"--------------Games list not up to date for todays games!!! Scraping disabled until list is updated.--------------")
                print(Style.RESET_ALL)
                odds = None
            else:
                print(f"------------------{args.odds} odds data------------------")
                for g in odds.keys():
                    home_team, away_team = g.split(":")
                    print(f"{away_team} ({odds[g][away_team]['money_line_odds']}) @ {home_team} ({odds[g][home_team]['money_line_odds']})")
        else:
            data = get_todays_games_json(todays_games_url)
            games = create_todays_games(data)
            
    data = get_json_data(data_url)
    df = to_data_frame(data)
    data, todays_games_uo, frame_ml, home_team_odds, away_team_odds = createTodaysGames(games, df, odds, use_test_data)
    
    if args.nn:
        print("------------Neural Network Model Predictions-----------")
        data = tf.keras.utils.normalize(data, axis=1)
        NN_Runner.nn_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc)
        print("-------------------------------------------------------")
    if args.xgb:
        print("---------------XGBoost Model Predictions---------------")
        XGBoost_Runner.xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc)
        print("-------------------------------------------------------")
    if args.A:
        print("---------------XGBoost Model Predictions---------------")
        XGBoost_Runner.xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc)
        print("-------------------------------------------------------")
        data = tf.keras.utils.normalize(data, axis=1)
        print("------------Neural Network Model Predictions-----------")
        NN_Runner.nn_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc)
        print("-------------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model to Run')
    parser.add_argument('-xgb', action='store_true', help='Run with XGBoost Model')
    parser.add_argument('-nn', action='store_true', help='Run with Neural Network Model')
    parser.add_argument('-A', action='store_true', help='Run all Models')
    parser.add_argument('-odds', help='Sportsbook to fetch from. (fanduel, draftkings, betmgm, pointsbet, caesars, wynn, bet_rivers_ny')
    parser.add_argument('-kc', action='store_true', help='Calculates percentage of bankroll to bet based on model edge')
    # テストデータ用の引数を追加
    parser.add_argument('-test', action='store_true', help='Use test data instead of API')
    args = parser.parse_args()
    main()