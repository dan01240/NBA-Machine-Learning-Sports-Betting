import argparse
import time
from datetime import datetime, timedelta

import pandas as pd
import tensorflow as tf
from colorama import Fore, Style

from src.DataProviders.SbrOddsProvider import SbrOddsProvider
from src.Predict import NN_Runner, XGBoost_Runner
from src.Utils.Dictionaries import team_index_current
from src.Utils.tools import create_todays_games_from_odds, get_json_data, to_data_frame, get_todays_games_json, create_todays_games

# python main.py -xgb -odds=bet365 -kc
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


def normalize_team_name(team_name):
    """
    チーム名を辞書キーと一致するように正規化します。
    
    引数:
        team_name: APIまたはオッズサービスからのチーム名文字列
        
    戻り値:
        team_index_currentのキーと一致する正規化されたチーム名
    """
    # 処理する一般的なバリエーション
    replacements = {
        "LA Clippers": "Los Angeles Clippers",
        "LA Lakers": "Los Angeles Lakers",
        # 必要に応じて他のバリエーションを追加
    }
    
    # 直接置換を適用
    if team_name in replacements:
        return replacements[team_name]
    
    # 部分一致を試みる
    for key in team_index_current.keys():
        # API名が辞書名を含むか、またはその逆の場合
        if team_name in key or key in team_name:
            return key
    
    # 一致が見つからない場合は元の名前を返す
    return team_name


def createTodaysGames(games, df, odds):
    match_data = []
    todays_games_uo = []
    home_team_odds = []
    away_team_odds = []

    home_team_days_rest = []
    away_team_days_rest = []
    
    # デバッグ用にDataFrameの情報を表示
    print(f"DataFrame形状: {df.shape}")
    print(f"DataFrameの最初の5行:")
    print(df.head())
    
    # デバッグ用に今日の試合からチーム名を表示
    print("今日の試合のチーム:")
    for game in games:
        print(f"ホーム: '{game[0]}', アウェイ: '{game[1]}'")
    
    # 比較のために辞書内のすべてのチームを表示
    print("team_index_currentのチーム:")
    for team in sorted(team_index_current.keys()):
        print(f"'{team}': {team_index_current[team]}")

    for game in games:
        home_team_original = game[0]
        away_team_original = game[1]
        
        # チーム名を正規化
        home_team = normalize_team_name(home_team_original)
        away_team = normalize_team_name(away_team_original)
        
        # 正規化が行われたら通知
        if home_team != home_team_original:
            print(f"ホームチーム名を正規化: '{home_team_original}' -> '{home_team}'")
        if away_team != away_team_original:
            print(f"アウェイチーム名を正規化: '{away_team_original}' -> '{away_team}'")
        
        # チームが辞書内にあるかチェック
        if home_team not in team_index_current:
            print(f"警告: ホームチーム '{home_team}' はteam_index_currentに見つかりません")
            continue
            
        if away_team not in team_index_current:
            print(f"警告: アウェイチーム '{away_team}' はteam_index_currentに見つかりません")
            continue
            
        # インデックスがDataFrameに対して有効かチェック
        home_index = team_index_current.get(home_team)
        away_index = team_index_current.get(away_team)
        
        if home_index is None:
            print(f"エラー: ホームチーム '{home_team}' のインデックスが見つかりません")
            continue
            
        if away_index is None:
            print(f"エラー: アウェイチーム '{away_team}' のインデックスが見つかりません")
            continue
        
        if home_index >= df.shape[0]:
            print(f"エラー: ホームチームのインデックス {home_index} は {df.shape[0]} 行のDataFrameの範囲外です")
            continue
            
        if away_index >= df.shape[0]:
            print(f"エラー: アウェイチームのインデックス {away_index} は {df.shape[0]} 行のDataFrameの範囲外です")
            continue
        
        # オッズデータの処理
        if odds is not None:
            game_key = home_team + ':' + away_team
            if game_key not in odds:
                print(f"警告: 試合 {game_key} はオッズデータに見つかりません")
                # オリジナルのキーも試す
                game_key_original = home_team_original + ':' + away_team_original
                if game_key_original in odds:
                    print(f"  - オリジナルキー '{game_key_original}' で見つかりました")
                    game_key = game_key_original
                else:
                    # キーの一部が一致する可能性を探す
                    found = False
                    for key in odds.keys():
                        if home_team in key and away_team in key:
                            print(f"  - 部分一致キー '{key}' で見つかりました")
                            game_key = key
                            found = True
                            break
                    if not found:
                        continue

            game_odds = odds[game_key]
            todays_games_uo.append(game_odds['under_over_odds'])

            try:
                home_odds_key = home_team if home_team in game_odds else home_team_original
                away_odds_key = away_team if away_team in game_odds else away_team_original
                
                home_team_odds.append(game_odds[home_odds_key]['money_line_odds'])
                away_team_odds.append(game_odds[away_odds_key]['money_line_odds'])
            except KeyError as e:
                print(f"オッズデータの処理中にキーエラー: {str(e)}")
                print(f"利用可能なキー: {list(game_odds.keys())}")
                home_team_odds.append(None)
                away_team_odds.append(None)
        else:
            todays_games_uo.append(input(home_team + ' vs ' + away_team + ': '))
            home_team_odds.append(input(home_team + ' odds: '))
            away_team_odds.append(input(away_team + ' odds: '))

        try:
            # CSVファイルの読み込みをスキップし、固定値を使用する
            print(f"{away_team} 休日数: 1 @ {home_team} 休日数: 1")
            home_days_off = timedelta(days=1)
            away_days_off = timedelta(days=1)
            
            home_team_days_rest.append(1)
            away_team_days_rest.append(1)
            
            # チームデータの取得
            try:
                home_team_series = df.iloc[team_index_current.get(home_team)]
                away_team_series = df.iloc[team_index_current.get(away_team)]
                stats = pd.concat([home_team_series, away_team_series])
                stats['Days-Rest-Home'] = 1  # 固定値を使用
                stats['Days-Rest-Away'] = 1  # 固定値を使用
                match_data.append(stats)
            except Exception as e:
                print(f"チームデータの取得中にエラー: {str(e)}")
                raise
        except Exception as e:
            print(f"試合 {home_team} vs {away_team} の処理中にエラー: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # フィルタリング後に試合があるかチェック
    if not match_data:
        error_msg = "フィルタリング後に有効な試合がありません。チーム名とインデックスを確認してください。"
        print(error_msg)
        raise ValueError(error_msg)
        
    games_data_frame = pd.concat(match_data, ignore_index=True, axis=1)
    games_data_frame = games_data_frame.T

    # TEAM_IDとTEAM_NAMEが列に存在するか確認
    columns_to_drop = [col for col in ['TEAM_ID', 'TEAM_NAME'] if col in games_data_frame.columns]
    if columns_to_drop:
        frame_ml = games_data_frame.drop(columns=columns_to_drop)
    else:
        print("警告: 'TEAM_ID'または'TEAM_NAME'列が見つかりません")
        frame_ml = games_data_frame

    data = frame_ml.values
    data = data.astype(float)

    return data, todays_games_uo, frame_ml, home_team_odds, away_team_odds


def main():
    try:
        odds = None
        if args.odds:
            try:
                print(f"{args.odds}からオッズデータを取得しています...")
                odds = SbrOddsProvider(sportsbook=args.odds).get_odds()
                
                if not odds:
                    print(f"警告: {args.odds}からオッズデータを取得できませんでした")
                    return False
                    
                games = create_todays_games_from_odds(odds)
                if len(games) == 0:
                    print("試合が見つかりません。")
                    return False
                    
                if (games[0][0] + ':' + games[0][1]) not in list(odds.keys()):
                    print(games[0][0] + ':' + games[0][1])
                    print(Fore.RED, "--------------試合リストが今日の試合のために更新されていません!!! リストが更新されるまでスクレイピングは無効化されています.--------------")
                    print(Style.RESET_ALL)
                    odds = None
                else:
                    # オッズデータの表示
                    print(f"------------------{args.odds} オッズデータ------------------")
                    for g in odds.keys():
                        home_team, away_team = g.split(":")
                        print(f"{away_team} ({odds[g][away_team]['money_line_odds']}) @ {home_team} ({odds[g][home_team]['money_line_odds']})")
            except Exception as e:
                print(f"オッズデータの取得中にエラー: {str(e)}")
                import traceback
                traceback.print_exc()
                odds = None
        else:
            print("オッズが指定されていないため、todays_games_jsonからデータを取得しています...")
            try:
                data = get_todays_games_json(todays_games_url)
                games = create_todays_games(data)
                if len(games) == 0:
                    print("試合が見つかりません。")
                    return False
            except Exception as e:
                print(f"試合データの取得中にエラー: {str(e)}")
                import traceback
                traceback.print_exc()
                return False
        
        # NBA統計データの取得
        print("NBA統計データを取得しています...")
        try:
            data = get_json_data(data_url)
            if not data:
                print("NBA統計データを取得できませんでした。")
                return False
                
            df = to_data_frame(data)
            if df.empty:
                print("空のデータフレームが返されました。")
                return False
        except Exception as e:
            print(f"NBA統計データの取得中にエラー: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
        # 今日の試合の作成
        print("今日の試合データを準備しています...")
        try:
            data, todays_games_uo, frame_ml, home_team_odds, away_team_odds = createTodaysGames(games, df, odds)
        except Exception as e:
            print(f"今日の試合データの準備中にエラー: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
        # モデル予測の実行
        if args.nn:
            try:
                print("------------Neural Network Model Predictions-----------")
                data = tf.keras.utils.normalize(data, axis=1)
                NN_Runner.nn_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc)
                print("-------------------------------------------------------")
            except Exception as e:
                print(f"NNモデル実行中にエラー: {str(e)}")
                import traceback
                traceback.print_exc()
        
        if args.xgb:
            try:
                print("---------------XGBoost Model Predictions---------------")
                XGBoost_Runner.xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc)
                print("-------------------------------------------------------")
            except Exception as e:
                print(f"XGBoostモデル実行中にエラー: {str(e)}")
                import traceback
                traceback.print_exc()
        
        if args.A:
            try:
                print("---------------XGBoost Model Predictions---------------")
                XGBoost_Runner.xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc)
                print("-------------------------------------------------------")
                
                data = tf.keras.utils.normalize(data, axis=1)
                print("------------Neural Network Model Predictions-----------")
                NN_Runner.nn_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc)
                print("-------------------------------------------------------")
            except Exception as e:
                print(f"モデル実行中にエラー: {str(e)}")
                import traceback
                traceback.print_exc()
        
        return True
    except IndexError as e:
        print(f"エラー: インデックスエラーが発生しました: {str(e)}")
        print("これはAPIデータとteam_index_current辞書のチーム名の不一致が原因である可能性があります。")
        print("詳細については上記のデバッグ出力を確認してください。")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"エラー: 予期しないエラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='実行するモデル')
    parser.add_argument('-xgb', action='store_true', help='XGBoostモデルで実行')
    parser.add_argument('-nn', action='store_true', help='ニューラルネットワークモデルで実行')
    parser.add_argument('-A', action='store_true', help='すべてのモデルを実行')
    parser.add_argument('-odds', help='取得するブックメーカー（fanduel, draftkings, betmgm, pointsbet, caesars, wynn, bet_rivers_ny')
    parser.add_argument('-kc', action='store_true', help='モデルのエッジに基づいて賭けるバンクロールの割合を計算')
    args = parser.parse_args()
    success = main()
    if not success:
        import sys
        sys.exit(1)