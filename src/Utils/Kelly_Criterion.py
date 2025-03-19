# src/Utils/Kelly_Criterion.py

from src.Utils.Adaptive_Calibration import adjust_probability

def american_to_decimal(american_odds):
    """
    アメリカンオッズを小数オッズに変換する関数
    
    Args:
        american_odds: アメリカンオッズ
        
    Returns:
        decimal_odds: 小数オッズ
    """
    if american_odds is None:
        return 1.0
        
    if american_odds >= 100:
        decimal_odds = (american_odds / 100) + 1
    else:
        decimal_odds = (100 / abs(american_odds)) + 1
    return round(decimal_odds, 2)

def calculate_kelly_criterion(american_odds, model_prob):
    """
    Kelly Criterionを計算する関数
    
    Args:
        american_odds: アメリカンオッズ
        model_prob: モデルの予測確率
        
    Returns:
        bankroll_fraction: 賭けるべき資金の割合 (%)
    """
    if american_odds is None:
        return 0
    
    # Adaptive Calibrationで予測確率を調整
    adjusted_prob = adjust_probability(model_prob)
    
    decimal_odds = american_to_decimal(american_odds)
    # ケリー式: (p(b+1)-1)/b = p - (1-p)/b
    # p: 調整後の予測確率
    # b: 小数オッズ - 1
    b = decimal_odds - 1
    
    if b <= 0:
        return 0
    
    # 調整された確率を使ってKelly Criterionを計算
    bankroll_fraction = round((100 * (adjusted_prob - (1 - adjusted_prob) / b)), 2)
    
    # Kelly Criterionが負の場合は0を返す
    return bankroll_fraction if bankroll_fraction > 0 else 0