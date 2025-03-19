from src.Utils.Adaptive_Calibration import adjust_probability
def expected_value(Pwin, odds):
    Ploss = 1 - Pwin
    Mwin = payout(odds)
    return round((Pwin * Mwin) - (Ploss * 100), 2)


def payout(odds):
    if odds > 0:
        return odds
    else:
        return (100 / (-1 * odds)) * 100

# src/Utils/Expected_Value.py
def expected_value(Pwin, odds):
    """
    期待値を計算する関数
    
    Args:
        Pwin: モデルが予測する勝率
        odds: アメリカンオッズ
        
    Returns:
        ev: 期待値
    """
    # Adaptive Calibrationで予測確率を調整
    adjusted_Pwin = adjust_probability(Pwin)
    
    Ploss = 1 - adjusted_Pwin
    Mwin = payout(odds)
    return round((adjusted_Pwin * Mwin) - (Ploss * 100), 2)


def payout(odds):
    """
    オッズから配当を計算する関数
    
    Args:
        odds: アメリカンオッズ
        
    Returns:
        payout: $100賭けた場合の利益
    """
    if odds is None:
        return 0
        
    if odds > 0:
        return odds
    else:
        return (100 / (-1 * odds)) * 100