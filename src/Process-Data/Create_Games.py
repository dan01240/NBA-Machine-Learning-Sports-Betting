import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import toml

sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from src.Utils.Dictionaries import team_index_07, team_index_08, team_index_12, team_index_13, team_index_14, \
    team_index_current

config = toml.load("../../config.toml")

df = pd.DataFrame
scores = []
win_margin = []
OU = []
OU_Cover = []
games = []
days_rest_away = []
days_rest_home = []
teams_con = sqlite3.connect("../../Data/TeamData.sqlite")
odds_con = sqlite3.connect("../../Data/OddsData.sqlite")

# 処理できたゲーム数
processed_games = 0
skipped_games = 0

# 2019年以降のシーズンのみを処理対象とする
# 2012年から現在までの全シーズンのリスト
target_seasons = [
    '2012-13',  # 最初のシーズン
    '2013-14', 
    '2014-15',
    '2015-16',
    '2016-17',
    '2017-18',
    '2018-19',
    '2019-20',  # COVID-19で中断されたシーズン
    '2020-21',  # バブルシーズン
    '2021-22',  # ほぼ通常運営に戻ったシーズン
    '2022-23',  # 通常シーズン
    '2023-24',  # 現在のシーズン
    '2024-25'   # 次のシーズン（まだデータがない可能性あり）
]
print(f"対象シーズン: {', '.join(target_seasons)}")

for key, value in config['create-games'].items():
    # 対象シーズンのみ処理する
    if key not in target_seasons:
        print(f"シーズン {key} はスキップします（2019年以前のデータ）")
        continue
        
    print(f"シーズン {key} の処理を開始します...")
    try:
        odds_df = pd.read_sql_query(f"select * from \"odds_{key}_new\"", odds_con, index_col="index")
        print(f"オッズデータ: {len(odds_df)}試合を読み込みました")
        
        team_table_str = key
        year_count = 0
        season = key

        for row in odds_df.itertuples():
            try:
                home_team = row[2]
                away_team = row[3]
                date = row[1]

                # TeamDataにテーブルが存在するか確認
                try:
                    table_check = pd.read_sql_query(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{date}'", teams_con)
                    if len(table_check) == 0:
                        print(f"  日付 {date} のチームデータが見つかりません。この試合をスキップします。")
                        skipped_games += 1
                        continue
                        
                    team_df = pd.read_sql_query(f"select * from \"{date}\"", teams_con, index_col="index")
                except Exception as e:
                    print(f"  日付 {date} のチームデータ読み込みエラー: {e}")
                    skipped_games += 1
                    continue
                    
                if len(team_df.index) == 30:
                    scores.append(row[8])
                    OU.append(row[4])
                    days_rest_home.append(row[10])
                    days_rest_away.append(row[11])
                    if row[9] > 0:
                        win_margin.append(1)
                    else:
                        win_margin.append(0)

                    if row[8] < row[4]:
                        OU_Cover.append(0)
                    elif row[8] > row[4]:
                        OU_Cover.append(1)
                    elif row[8] == row[4]:
                        OU_Cover.append(2)

                    # チームインデックスを取得
                    try:
                        if season == '2007-08':
                            home_team_series = team_df.iloc[team_index_07.get(home_team)]
                            away_team_series = team_df.iloc[team_index_07.get(away_team)]
                        elif season == '2008-09' or season == "2009-10" or season == "2010-11" or season == "2011-12":
                            home_team_series = team_df.iloc[team_index_08.get(home_team)]
                            away_team_series = team_df.iloc[team_index_08.get(away_team)]
                        elif season == "2012-13":
                            home_team_series = team_df.iloc[team_index_12.get(home_team)]
                            away_team_series = team_df.iloc[team_index_12.get(away_team)]
                        elif season == '2013-14':
                            home_team_series = team_df.iloc[team_index_13.get(home_team)]
                            away_team_series = team_df.iloc[team_index_13.get(away_team)]
                        elif season == '2022-23' or season == '2023-24' or season == '2024-25':
                            home_team_series = team_df.iloc[team_index_current.get(home_team)]
                            away_team_series = team_df.iloc[team_index_current.get(away_team)]
                        else:
                            home_team_series = team_df.iloc[team_index_14.get(home_team)]
                            away_team_series = team_df.iloc[team_index_14.get(away_team)]
                            
                        game = pd.concat([home_team_series, away_team_series.rename(
                            index={col: f"{col}.1" for col in team_df.columns.values}
                        )])
                        games.append(game)
                        processed_games += 1
                        
                        if processed_games % 50 == 0:
                            print(f"  {processed_games}試合処理しました...")
                    except Exception as e:
                        print(f"  チームインデックスエラー ({home_team} vs {away_team}): {e}")
                        skipped_games += 1
                        continue
                else:
                    print(f"  日付 {date} のチームデータが不完全です（{len(team_df.index)}チーム）")
                    skipped_games += 1
            except Exception as e:
                print(f"  試合データ処理エラー: {e}")
                skipped_games += 1
                continue
    except Exception as e:
        print(f"シーズン {key} のオッズデータ読み込みエラー: {e}")
        continue

odds_con.close()
teams_con.close()

print(f"\n処理結果サマリー:")
print(f"処理した試合: {processed_games}")
print(f"スキップした試合: {skipped_games}")

if processed_games == 0:
    print("処理できた試合がありません。データセットは作成されませんでした。")
    sys.exit(1)

print("\nデータセットを作成しています...")
season = pd.concat(games, ignore_index=True, axis=1)
season = season.T
frame = season.drop(columns=['TEAM_ID', 'TEAM_ID.1'])
frame['Score'] = np.asarray(scores)
frame['Home-Team-Win'] = np.asarray(win_margin)
frame['OU'] = np.asarray(OU)
frame['OU-Cover'] = np.asarray(OU_Cover)
frame['Days-Rest-Home'] = np.asarray(days_rest_home)
frame['Days-Rest-Away'] = np.asarray(days_rest_away)

# 型の修正
for field in frame.columns.values:
    if 'TEAM_' in field or 'Date' in field or field not in frame:
        continue
    frame[field] = frame[field].astype(float)

# データベースに保存（別名で保存）
con = sqlite3.connect("../../Data/dataset.sqlite")
frame.to_sql("dataset_2012-25_test", con, if_exists="replace")
con.close()

print("2019-2025年のデータセット（dataset_2019-25）の作成が完了しました！")
print(f"合計 {processed_games} 試合のデータを処理しました。")