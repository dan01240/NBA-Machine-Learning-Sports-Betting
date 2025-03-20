import sqlite3
from datetime import datetime
import pandas as pd
import toml

# データベース接続
odds_con = sqlite3.connect("../../Data/OddsData.sqlite")
config = toml.load("../../config.toml")

# 現在のデータ構造を確認するための関数
def inspect_table(table_name):
    try:
        # テーブルが存在するか確認
        check_df = pd.read_sql_query(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'", odds_con)
        if len(check_df) == 0:
            print(f"テーブル '{table_name}' は存在しません")
            return False
            
        # サンプルデータを取得
        sample_df = pd.read_sql_query(f"SELECT * FROM \"{table_name}\" LIMIT 5", odds_con)
        print(f"\nテーブル '{table_name}' のサンプルデータ:")
        print(sample_df)
        
        # カラム情報を表示
        print(f"\nカラム情報:")
        for col in sample_df.columns:
            print(f"  - {col}: {sample_df[col].dtype}")
            
        return True
    except Exception as e:
        print(f"テーブル '{table_name}' の確認中にエラーが発生しました: {e}")
        return False

# シンプルに_newテーブルを作成する関数
def create_new_table(source_table, target_table):
    try:
        # 元のテーブルをそのままコピー
        df = pd.read_sql_query(f"SELECT * FROM \"{source_table}\"", odds_con, index_col="index")
        print(f"テーブル '{source_table}' から {len(df)} 行のデータを読み込みました")
        
        # 日付カラムの確認
        if 'Date' in df.columns:
            print(f"日付サンプル: {df['Date'].iloc[0:5].tolist()}")
            
            # 既に適切な形式かどうかを確認
            if isinstance(df['Date'].iloc[0], str) and df['Date'].iloc[0].count('-') == 2:
                print("日付は既に適切な形式 (YYYY-MM-DD) のようです")
                # そのまま保存
                df.to_sql(target_table, odds_con, if_exists="replace")
                print(f"テーブル '{target_table}' を作成しました")
                return True
        
        # その他のカラムも確認
        print("テーブルのカラム:")
        for col in df.columns:
            print(f"  - {col}")
        
        # そのまま保存
        df.to_sql(target_table, odds_con, if_exists="replace")
        print(f"テーブル '{target_table}' を作成しました")
        return True
        
    except Exception as e:
        print(f"テーブル '{source_table}' から '{target_table}' の作成中にエラーが発生しました: {e}")
        return False

# メインの処理
print("オッズデータのテーブル処理を開始します...\n")

# 処理対象を指定（ここでは2024-25のみを処理）
target_seasons = ['2024-25']

for season in target_seasons:
    source_table = f"odds_{season}"
    target_table = f"odds_{season}_new"
    
    print(f"\n--- {season} シーズンの処理 ---")
    
    # テーブル構造を確認
    if inspect_table(source_table):
        # 新しいテーブルを作成
        create_new_table(source_table, target_table)
    else:
        print(f"{source_table} テーブルが見つかりません。スキップします。")

odds_con.close()
print("\n処理が完了しました。")