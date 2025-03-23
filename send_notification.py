#!/usr/bin/env python3
import os
import sys
import subprocess
import requests
import json
import traceback
from datetime import datetime

def send_slack_notification(message, webhook_url):
    """Slackに通知を送信する"""
    print(f"Slack通知を送信中... メッセージ長: {len(message)} 文字")
    try:
        # 長すぎる場合はSlackの制限に合わせて分割
        max_length = 40000  # Slackの1メッセージの上限に近い値

        if len(message) <= max_length:
            print("メッセージを単一のブロックとして送信")
            blocks = [{
                "type": "section",
                "text": {
                    "type": "plain_text",
                    "text": message[:3000] + "..." if len(message) > 3000 else message
                }
            }]
            payload = {"text": message}  # Simple text fallback
            print("POSTリクエストを送信")
            response = requests.post(
                webhook_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"}
            )
            print(f"レスポンスステータス: {response.status_code}")
            print(f"レスポンス内容: {response.text[:100]}")
            if response.status_code != 200:
                raise ValueError(f"Slack通知の送信に失敗: {response.status_code}, {response.text}")
        else:
            # メッセージが長い場合は分割して送信
            print(f"メッセージが長いため分割して送信します: {len(message)} 文字")
            parts = [message[i:i+max_length] for i in range(0, len(message), max_length)]
            print(f"{len(parts)}個のパートに分割しました")
            
            for i, part in enumerate(parts, 1):
                print(f"パート {i}/{len(parts)} を送信中...")
                payload = {"text": f"NBA ベッティング予測 {datetime.now().strftime('%Y-%m-%d %H:%M')} (パート {i}/{len(parts)})\n\n{part}"}
                response = requests.post(
                    webhook_url,
                    data=json.dumps(payload),
                    headers={"Content-Type": "application/json"}
                )
                print(f"パート {i} レスポンスステータス: {response.status_code}")
                if response.status_code != 200:
                    raise ValueError(f"Slack通知の送信に失敗: {response.status_code}, {response.text}")
        
        return True
    except Exception as e:
        print(f"Slack通知の送信中にエラーが発生しました: {str(e)}")
        traceback.print_exc()
        return False

def run_betting_model():
    """ベッティングモデルを実行して結果を取得"""
    print("ベッティングモデルの実行を開始...")
    try:
        # コマンドを実行して出力を取得
        print("subprocess.run を呼び出し中...")
        result = subprocess.run(
            ['python', 'main.py', '-xgb', '-odds=bet365', '-kc'], 
            capture_output=True, 
            text=True,
            check=False  # エラーが発生してもスクリプトを継続
        )
        
        if result.returncode != 0:
            print(f"ベッティングモデル実行エラー (コード {result.returncode})")
            print(f"標準エラー出力: {result.stderr}")
            return f"エラー: ベッティングモデルの実行に失敗しました (コード {result.returncode})。\n\n標準エラー出力:\n{result.stderr}\n\n標準出力:\n{result.stdout}"
        
        print(f"ベッティングモデル実行成功。出力: {len(result.stdout)} 文字")
        return result.stdout
    except Exception as e:
        print(f"ベッティングモデル実行中に例外が発生: {str(e)}")
        traceback.print_exc()
        return f"エラー: ベッティングモデル実行中に例外が発生しました。\n{str(e)}"

def main():
    print(f"=== スクリプト実行開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"Python バージョン: {sys.version}")
    print(f"実行パス: {os.getcwd()}")
    print(f"コマンドライン引数: {sys.argv}")
    
    # Slack webhookを取得
    slack_webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not slack_webhook_url:
        print("エラー: SLACK_WEBHOOK_URL環境変数が設定されていません")
        sys.exit(1)
    else:
        print("SLACK_WEBHOOK_URL環境変数が設定されています")
    
    try:
        # モデルを実行する場合
        if len(sys.argv) > 1 and sys.argv[1] == "--run-model":
            print("--run-model オプションが指定されています")
            print("ベッティングモデルを実行します...")
            message = run_betting_model()
        # すでに出力が存在する場合（標準入力から読み込み）
        elif not sys.stdin.isatty():
            print("標準入力からメッセージを読み込み中...")
            message = sys.stdin.read()
            print(f"標準入力から {len(message)} 文字を読み込みました")
        # コマンドライン引数から読み込み
        elif len(sys.argv) > 1:
            print(f"ファイル '{sys.argv[1]}' からメッセージを読み込み中...")
            try:
                with open(sys.argv[1], 'r') as f:
                    message = f.read()
                print(f"ファイルから {len(message)} 文字を読み込みました")
            except Exception as e:
                print(f"ファイル読み込みエラー: {str(e)}")
                message = f"ファイル {sys.argv[1]} の読み込み中にエラーが発生しました: {str(e)}"
        else:
            print("使用法: python send_notification.py [ファイルパス | --run-model]")
            print("または: cat 出力ファイル | python send_notification.py")
            sys.exit(1)
        
        if not message or len(message.strip()) == 0:
            print("警告: メッセージが空です")
            message = "警告: ベッティングモデルからの出力が空でした。"
        
        print("Slack通知を送信します...")
        if send_slack_notification(message, slack_webhook_url):
            print("Slack通知を送信しました")
        else:
            print("Slack通知の送信に失敗しました")
            sys.exit(1)
    
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {str(e)}")
        traceback.print_exc()
        try:
            error_message = f"NBA ベッティング実行中にエラーが発生しました:\n\n{str(e)}\n\n{traceback.format_exc()}"
            send_slack_notification(error_message, slack_webhook_url)
        except:
            print("エラー通知の送信にも失敗しました")
        sys.exit(1)
    
    print(f"=== スクリプト実行終了: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

if __name__ == "__main__":
    main()