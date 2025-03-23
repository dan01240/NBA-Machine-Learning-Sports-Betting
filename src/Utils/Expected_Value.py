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

# # src/Utils/Expected_Value.py
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

# def expected_value(Pwin, odds):
#     """
#     モデルの平均キャリブレーション誤差を考慮した期待値計算
#     """
#     # 予測範囲に応じた平均キャリブレーション誤差の適用
#     if Pwin > 0.7:
#         # 高確率予測には大きめの誤差
#         cal_error = 0.038  # 3.8%
#     elif Pwin > 0.6:
#         # 中程度の確率には標準誤差
#         cal_error = 0.035  # 3.5%
#     else:
#         # 低確率予測には小さめの誤差
#         cal_error = 0.030  # 3.0%
    
#     # キャリブレーション誤差を引いて調整
#     adjusted_Pwin = max(0.01, min(0.99, Pwin - cal_error))
    
#     Ploss = 1 - adjusted_Pwin
#     Mwin = payout(odds)
#     return round((adjusted_Pwin * Mwin) - (Ploss * 100), 2)


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