import re
import time  # Add this import
from datetime import datetime

import pandas as pd
import requests

from .Dictionaries import team_index_current

games_header = {
    'user-agent': 'Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/57.0.2987.133 Safari/537.36',
    'Dnt': '1',
    'Accept-Encoding': 'gzip, deflate, sdch',
    'Accept-Language': 'en',
    'origin': 'http://stats.nba.com',
    'Referer': 'https://github.com'
}

data_headers = {
    'Accept': 'application/json, text/plain, */*',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9',
    'Connection': 'keep-alive',
    'Host': 'stats.nba.com',
    'Origin': 'https://www.nba.com',
    'Referer': 'https://www.nba.com/',
    'sec-ch-ua': '"Google Chrome";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-site',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'x-nba-stats-origin': 'stats',
    'x-nba-stats-token': 'true'
}


def get_json_data(url, max_retries=3, backoff_factor=1.5):
    """
    Fetch JSON data from the NBA API with retry logic
    
    Args:
        url: API endpoint URL
        max_retries: Maximum number of retry attempts
        backoff_factor: Factor to increase wait time between retries
    
    Returns:
        JSON data or empty dict on failure
    """
    retries = 0
    last_exception = None
    
    while retries < max_retries:
        try:
            print(f"Attempting request to {url} (Attempt {retries + 1}/{max_retries})")
            raw_data = requests.get(url, headers=data_headers, timeout=30)
            raw_data.raise_for_status()  # Raise exception for 4XX/5XX responses
            json = raw_data.json()
            return json.get('resultSets')
        except Exception as e:
            print(f"Request failed: {str(e)}")
            last_exception = e
            retries += 1
            if retries < max_retries:
                wait_time = backoff_factor ** retries
                print(f"Waiting {wait_time:.1f} seconds before retry...")
                time.sleep(wait_time)
    
    # If we get here, all retries failed
    print(f"All {max_retries} attempts failed. Last error: {str(last_exception)}")
    return {}



def get_todays_games_json(url):
    raw_data = requests.get(url, headers=games_header)
    json = raw_data.json()
    return json.get('gs').get('g')


def to_data_frame(data):
    try:
        data_list = data[0]
    except Exception as e:
        print(e)
        return pd.DataFrame(data={})
    return pd.DataFrame(data=data_list.get('rowSet'), columns=data_list.get('headers'))


def create_todays_games(input_list):
    games = []
    for game in input_list:
        home = game.get('h')
        away = game.get('v')
        home_team = home.get('tc') + ' ' + home.get('tn')
        away_team = away.get('tc') + ' ' + away.get('tn')
        games.append([home_team, away_team])
    return games


def create_todays_games_from_odds(input_dict):
    games = []
    for game in input_dict.keys():
        home_team, away_team = game.split(":")
        if home_team not in team_index_current or away_team not in team_index_current:
            continue
        games.append([home_team, away_team])
    return games


def get_date(date_string):
    year1, month, day = re.search(r'(\d+)-\d+-(\d\d)(\d\d)', date_string).groups()
    year = year1 if int(month) > 8 else int(year1) + 1
    return datetime.strptime(f"{year}-{month}-{day}", '%Y-%m-%d')
