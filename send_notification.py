#!/usr/bin/env python3
import os
import sys
import subprocess
import requests
import json
from datetime import datetime

def send_slack_notification(message, webhook_url):
    """Slackに通知を送信する"""
    try:
        # 長すぎる場合はSlackの制限に合わせて分割
        max_length = 40000  # Slackの1メッセージの上限に近い値

        if len(message) <= max_length:
            blocks = [{
                "type": "section",
                "text": {
                    "type": "plain_text",
                    "text": message
                }
            }]
            payload = {"blocks": blocks}
            response = requests.post(
                webhook_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"}
            )
            if response.status_code != 200:
                raise ValueError(f"Slack通知の送信に失敗: {response.status_code}, {response.text}")
        else:
            # メッセージが長い場合は分割して送信
            parts = [message[i:i+max_length] for i in range(0, len(message), max_length)]
            
            # 最初のパートを送信
            blocks = [{
                "type": "section",
                "text": {
                    "type": "plain_text",
                    "text": f"NBA ベッティング予測 {datetime.now().strftime('%Y-%m-%d %H:%M')} (1/{len(parts)})"
                }
            }, {
                "type": "section",
                "text": {
                    "type": "plain_text",
                    "text": parts[0]
                }
            }]
            payload = {"blocks": blocks}
            response = requests.post(
                webhook_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"}
            )
            
            # 残りのパートを送信
            for i, part in enumerate(parts[1:], 2):
                blocks = [{
                    "type": "section",
                    "text": {
                        "type": "plain_text",
                        "text": f"NBA ベッティング予測 続き ({i}/{len(parts)})"
                    }
                }, {
                    "type": "section",
                    "text": {
                        "type": "plain_text",
                        "text": part
                    }
                }]
                payload = {"blocks": blocks}
                response = requests.post(
                    webhook_url,
                    data=json.dumps(payload),
                    headers={"Content-Type": "application/json"}
                )
        
        return True
    except Exception as e:
        print(f"Slack通知の送信中にエラーが発生しました: {str(e)}")
        return False

def run_betting_model():
    """ベッティングモデルを実行して結果を取得"""
    try:
        # コマンドを実行して出力を取得
        result = subprocess.run(
            ['python', 'main.py', '-xgb', '-odds=bet365', '-kc'], 
            capture_output=True, 
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"エラー: ベッティングモデルの実行に失敗しました。\n{e.stderr}"
    except Exception as e:
        return f"エラー: {str(e)}"

def main():
    # Slack webhookを取得
    slack_webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not slack_webhook_url:
        print("エラー: SLACK_WEBHOOK_URL環境変数が設定されていません")
        sys.exit(1)
    
    # モデルを実行する場合
    if len(sys.argv) > 1 and sys.argv[1] == "--run-model":
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M')
        print(f"[{current_time}] ベッティングモデルを実行中...")
        message = run_betting_model()
    # すでに出力が存在する場合（標準入力から読み込み）
    elif not sys.stdin.isatty():
        message = sys.stdin.read()
    # コマンドライン引数から読み込み
    elif len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            message = f.read()
    else:
        print("使用法: python send_notification.py [ファイルパス | --run-model]")
        print("または: cat 出力ファイル | python send_notification.py")
        sys.exit(1)
    
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M')
    print(f"[{current_time}] Slack通知を送信中...")
    if send_slack_notification(message, slack_webhook_url):
        print(f"[{current_time}] Slack通知を送信しました")
    else:
        print(f"[{current_time}] Slack通知の送信に失敗しました")
        print("予測結果:")
        print(message)

if __name__ == "__main__":
    main()