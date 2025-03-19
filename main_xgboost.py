# run_xgboost.py
import pandas as pd
from datetime import datetime, timedelta
from colorama import Fore, Style

from src.DataProviders.SbrOddsProvider import SbrOddsProvider
from src.Predict import XGBoost_Runner
from src.Utils.Dictionaries import team_index_current
from src.Utils.tools import create_todays_games_from_odds, get_json_data, to_data_frame

# データURL
data_url = 'https://stats.nba.com/stats/leaguedashteamstats?' \
           'Conference=&DateFrom=&DateTo=&Division=&GameScope=&' \
           'GameSegment=&LastNGames=0&LeagueID=00&Location=&' \
           'MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&' \
           'PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&' \
           'PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&' \
           'Season=2024-25&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&' \
           'StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='

# オッズデータを取得
# sportsbook = 'fanduel'
# sportsbook = 'draftkings'
# sportsbook = 'betmgm'
# sportsbook = 'pointsbet'
# sportsbook = 'caesars'
# sportsbook = 'wynn'
sportsbook = 'bet365'

odds = SbrOddsProvider(sportsbook=sportsbook).get_odds()
games = create_todays_games_from_odds(odds)

if len(games) == 0:
    print("No games found.")
    exit()

if (games[0][0] + ':' + games[0][1]) not in list(odds.keys()):
    print(games[0][0] + ':' + games[0][1])
    print(Fore.RED, "--------------Games list not up to date for todays games!!! Scraping disabled until list is updated.--------------")
    print(Style.RESET_ALL)
    exit()
else:
    print(f"------------------{sportsbook} odds data------------------")
    for g in odds.keys():
        home_team, away_team = g.split(":")
        print(f"{away_team} ({odds[g][away_team]['money_line_odds']}) @ {home_team} ({odds[g][home_team]['money_line_odds']})")

# チームデータを取得
data = get_json_data(data_url)
df = to_data_frame(data)

# 試合データを作成
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
    
    game_odds = odds[home_team + ':' + away_team]
    todays_games_uo.append(game_odds['under_over_odds'])
    home_team_odds.append(game_odds[home_team]['money_line_odds'])
    away_team_odds.append(game_odds[away_team]['money_line_odds'])
    
    # 休息日数の計算
    try:
        schedule_df = pd.read_csv('Data/nba-2024-UTC.csv', parse_dates=['Date'], date_format='%d/%m/%Y %H:%M')
        home_games = schedule_df[(schedule_df['Home Team'] == home_team) | (schedule_df['Away Team'] == home_team)]
        away_games = schedule_df[(schedule_df['Home Team'] == away_team) | (schedule_df['Away Team'] == away_team)]
        previous_home_games = home_games.loc[schedule_df['Date'] <= datetime.today()].sort_values('Date', ascending=False).head(1)['Date']
        previous_away_games = away_games.loc[schedule_df['Date'] <= datetime.today()].sort_values('Date', ascending=False).head(1)['Date']
        
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
            
        print(f"{away_team} days off: {away_days_off.days} @ {home_team} days off: {home_days_off.days}")
        
        home_team_days_rest.append(home_days_off.days)
        away_team_days_rest.append(away_days_off.days)
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

# Noneチェックと修正
if all(odd is None for odd in home_team_odds):
    print("オッズデータが正しく取得できませんでした。仮のオッズデータを使用します。")
    home_team_odds = [-110] * len(games)
    away_team_odds = [-110] * len(games)

# XGBoostモデルを実行
print("---------------XGBoost Model Predictions---------------")
XGBoost_Runner.xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, True)
print("-------------------------------------------------------")