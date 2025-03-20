import os
import random
import sqlite3
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import toml
from sbrscrape import Scoreboard

# TODO: Add tests

sys.path.insert(1, os.path.join(sys.path[0], '../..'))

sportsbook = 'bet365'

# 相対パスが正しいことを確認
try:
    config = toml.load("../../config.toml")
    con = sqlite3.connect("../../Data/OddsData.sqlite")
    print("プロジェクトルートからの相対パスを使用")
except:
    try:
        config = toml.load("config.toml")
        con = sqlite3.connect("Data/OddsData.sqlite")
        print("カレントディレクトリからの相対パスを使用")
    except Exception as e:
        print(f"TOMLファイルまたはデータベースを開けませんでした: {e}")
        sys.exit(1)

total_games_saved = 0

for key, value in config['get-odds-data'].items():
    print(f"\n--- {key} シーズンのデータ取得を開始 ---")
    
    # 各シーズンごとにdf_dataをリセット
    df_data = []
    
    date_pointer = datetime.strptime(value['start_date'], "%Y-%m-%d").date()
    end_date = datetime.strptime(value['end_date'], "%Y-%m-%d").date()
    teams_last_played = {}

    while date_pointer <= end_date:
        print(f"オッズデータ取得中: {date_pointer}")
        sb = Scoreboard(date=date_pointer, sport="NBA")

        if not hasattr(sb, "games"):
            print(f"  {date_pointer}の試合データが見つかりません")
            date_pointer = date_pointer + timedelta(days=1)
            continue

        # 見つかった試合数を表示
        print(f"  {len(sb.games)}試合見つかりました")

        for game in sb.games:
            if game['home_team'] not in teams_last_played:
                teams_last_played[game['home_team']] = date_pointer
                home_games_rested = timedelta(days=7)  # start of season, big number
            else:
                current_date = date_pointer
                home_games_rested = current_date - teams_last_played[game['home_team']]
                teams_last_played[game['home_team']] = current_date

            if game['away_team'] not in teams_last_played:
                teams_last_played[game['away_team']] = date_pointer
                away_games_rested = timedelta(days=7)  # start of season, big number
            else:
                current_date = date_pointer
                away_games_rested = current_date - teams_last_played[game['away_team']]
                teams_last_played[game['away_team']] = current_date

            try:
                # オッズデータが存在するか確認
                if sportsbook in game['total'] and sportsbook in game['home_ml'] and sportsbook in game['away_ml'] and sportsbook in game['away_spread']:
                    df_data.append({
                        'Date': date_pointer,
                        'Home': game['home_team'],
                        'Away': game['away_team'],
                        'OU': game['total'][sportsbook],
                        'Spread': game['away_spread'][sportsbook],
                        'ML_Home': game['home_ml'][sportsbook],
                        'ML_Away': game['away_ml'][sportsbook],
                        'Points': game['away_score'] + game['home_score'],
                        'Win_Margin': game['home_score'] - game['away_score'],
                        'Days_Rest_Home': home_games_rested.days,
                        'Days_Rest_Away': away_games_rested.days
                    })
                else:
                    print(f"  {sportsbook}のオッズデータが不完全: {game['home_team']} vs {game['away_team']}")
            except KeyError as e:
                print(f"  {sportsbook}のオッズデータが見つかりません: {game['home_team']} vs {game['away_team']} - {e}")
            except Exception as e:
                print(f"  エラー: {e}")

        date_pointer = date_pointer + timedelta(days=1)
        time.sleep(random.randint(1, 3))

    # データが取得できたか確認
    if len(df_data) > 0:
        print(f"\n{key}シーズンの{len(df_data)}試合分のデータを保存します...")
        df = pd.DataFrame(df_data)
        
        # データベースに保存
        try:
            df.to_sql(f"odds_{key}", con, if_exists="replace")
            print(f"テーブル 'odds_{key}' にデータを保存しました")
            total_games_saved += len(df_data)
        except Exception as e:
            print(f"データベースへの保存中にエラーが発生しました: {e}")
    else:
        print(f"\n{key}シーズンのデータは見つかりませんでした")

con.close()
print(f"\n処理が完了しました。合計{total_games_saved}試合のデータを保存しました。")