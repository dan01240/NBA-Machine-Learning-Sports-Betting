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


def createTodaysGames(games, df, odds):
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
        if odds is not None:
            game_odds = odds[home_team + ':' + away_team]
            todays_games_uo.append(game_odds['under_over_odds'])

            home_team_odds.append(game_odds[home_team]['money_line_odds'])
            away_team_odds.append(game_odds[away_team]['money_line_odds'])

        else:
            todays_games_uo.append(input(home_team + ' vs ' + away_team + ': '))

            home_team_odds.append(input(home_team + ' odds: '))
            away_team_odds.append(input(away_team + ' odds: '))

        # calculate days rest for both teams
        schedule_df = pd.read_csv('Data/nba-2024-UTC.csv', parse_dates=['Date'], date_format='%d/%m/%Y %H:%M')
        home_games = schedule_df[(schedule_df['Home Team'] == home_team) | (schedule_df['Away Team'] == home_team)]
        away_games = schedule_df[(schedule_df['Home Team'] == away_team) | (schedule_df['Away Team'] == away_team)]
        previous_home_games = home_games.loc[schedule_df['Date'] <= datetime.today()].sort_values('Date',ascending=False).head(1)['Date']
        previous_away_games = away_games.loc[schedule_df['Date'] <= datetime.today()].sort_values('Date',ascending=False).head(1)['Date']
        if len(previous_home_games) > 0:
            last_home_date = previous_home_games.iloc[0]
            home_days_off = timedelta(days=1) + datetime.today() - last_home_date
        else:
            home_days_off = timedelta(days=7)
        if len(previous_away_games) > 0:
            last_away_date = previous_away_games.iloc[0]
            away_days_off = timedelta(days=1) + datetime.today() - last_away_date
        else:
            away_days_off = timedelta(days=7)
        # print(f"{away_team} days off: {away_days_off.days} @ {home_team} days off: {home_days_off.days}")

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
            # この部分はbet365から取得したオッズデータを表示しています。
            # Dallas Mavericks (375) @ Indiana Pacers (-500)
            # Houston Rockets (-135) @ Orlando Magic (115)
            # Detroit Pistons (-185) @ Miami Heat (155)
            # New Orleans Pelicans (575) @ Minnesota Timberwolves (-850)
            # New York Knicks (-390) @ San Antonio Spurs (310)
            # Philadelphia 76ers (475) @ Oklahoma City Thunder (-650)
            # Washington Wizards (155) @ Utah Jazz (-185)
            # Denver Nuggets (-130) @ Los Angeles Lakers (110)            
            print(f"------------------{args.odds} odds data------------------")
            for g in odds.keys():
                home_team, away_team = g.split(":")
                print(f"{away_team} ({odds[g][away_team]['money_line_odds']}) @ {home_team} ({odds[g][home_team]['money_line_odds']})")
    else:
        data = get_todays_games_json(todays_games_url)
        games = create_todays_games(data)
    data = get_json_data(data_url)
    df = to_data_frame(data)
    data, todays_games_uo, frame_ml, home_team_odds, away_team_odds = createTodaysGames(games, df, odds)
    if args.nn:
        print("------------Neural Network Model Predictions-----------")
        data = tf.keras.utils.normalize(data, axis=1)
        NN_Runner.nn_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc)
        print("-------------------------------------------------------")
    if args.xgb:
        print("---------------XGBoost Model Predictions---------------")
        # 勝敗予測:  
        # Indiana Pacers (61.20%): ペイサーズが61.20%の確率で勝利すると予測
        # Houston Rockets (51.60%): ロケッツが51.60%の確率で勝利すると予測      
            # Indiana Pacers (61.20000076293945%) vs Dallas Mavericks:
        
        # オーバー/アンダー予測:
        # UNDER 234.5 (60.60%): 合計得点が234.5点未満になる確率が60.60%
        # OVER 209.5 (88.20%): 合計得点が209.5点を超える確率が88.20%
            # UNDER 234.5 (60.599998474121094%)
        
        # Orlando Magic vs Houston Rockets (51.599998474121094%): OVER 209.5 (88.19999694824219%)
        # Miami Heat (51.79999923706055%) vs Detroit Pistons: UNDER 218.5 (51.099998474121094%)
        # Minnesota Timberwolves (80.5999984741211%) vs New Orleans Pelicans: OVER 228.5 (61.400001525878906%)
        # San Antonio Spurs (51.20000076293945%) vs New York Knicks: UNDER 227.5 (78.80000305175781%)
        # Oklahoma City Thunder (81.19999694824219%) vs Philadelphia 76ers: UNDER 227 (79.80000305175781%)
        # Utah Jazz vs Washington Wizards (52.70000076293945%): OVER 234 (74.30000305175781%)
        # Los Angeles Lakers (54.400001525878906%) vs Denver Nuggets: UNDER 234 (73.4000015258789%)
        # Sacramento Kings vs Cleveland Cavaliers (58.400001525878906%): OVER 236 (66.80000305175781%)
        # Phoenix Suns (66.5999984741211%) vs Chicago Bulls: OVER 234.5 (85.9000015258789%)
        # Portland Trail Blazers vs Memphis Grizzlies (61.900001525878906%): UNDER 235.5 (55.099998474121094%)        
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
    args = parser.parse_args()
    main()
