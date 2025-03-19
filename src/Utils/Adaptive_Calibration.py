# src/Utils/Adaptive_Calibration.py

def calculate_calibration_error(model_prob, default_error=0.05):
    """
    キャリブレーション誤差を計算または推定する関数
    現段階では固定値を返しますが、将来的には履歴データに基づいて計算できます
    
    Args:
        model_prob: モデルの予測確率
        default_error: デフォルトのキャリブレーション誤差
        
    Returns:
        calibration_error: キャリブレーション誤差の推定値
    """
    # 最初のシンプルな実装では固定値を返す
    # 高い確率予測ほど誤差が大きくなる傾向があるため、
    # 予測確率に比例して誤差を増加させる
    if model_prob > 0.7:
        # 高確率の予測には大きめの誤差
        return default_error * 1.5
    elif model_prob > 0.6:
        # 中程度の確率には標準誤差
        return default_error
    else:
        # 低確率の予測には小さめの誤差
        return default_error * 0.8

def adjust_probability(model_prob):
    """
    予測確率をAdaptive Calibrationで調整する関数
    
    Args:
        model_prob: モデルの予測確率
        
    Returns:
        adjusted_prob: 調整後の予測確率
    """
    calibration_error = calculate_calibration_error(model_prob)
    # キャリブレーション誤差の半分を差し引く
    adjusted_prob = max(0.01, min(0.99, model_prob - (calibration_error / 2)))
    return adjusted_prob

def adaptive_kelly(model_prob, odds, bankroll, prediction_history=None):
    """
    適応型ケリー基準を計算する関数
    
    Args:
        model_prob: モデルの予測確率
        odds: アメリカンオッズ
        bankroll: 総資金額
        prediction_history: 過去の予測結果のリスト（オプション）
        
    Returns:
        bet_amount: 推奨賭け金額
    """
    # 予測確率を調整
    adjusted_prob = adjust_probability(model_prob)
    
    # オッズをデシマル形式に変換
    if odds > 0:
        decimal_odds = odds / 100 + 1
    else:
        decimal_odds = 100 / abs(odds) + 1
    
    # ケリー式で計算
    edge = adjusted_prob * decimal_odds - 1
    if edge <= 0:
        return 0
    
    kelly_fraction = adjusted_prob - (1 - adjusted_prob) / (decimal_odds - 1)
    kelly_fraction = max(0, min(kelly_fraction, 0.25))  # 25%を上限に
    
    # 過去の予測結果に基づく調整（存在する場合）
    if prediction_history and len(prediction_history) > 0:
        # 過去の結果に基づいて信頼係数を計算（簡易版）
        confidence_factor = 0.5  # デフォルト値
        kelly_fraction *= confidence_factor
    
    # 賭け金額を計算
    bet_amount = bankroll * kelly_fraction
    
    return bet_amount