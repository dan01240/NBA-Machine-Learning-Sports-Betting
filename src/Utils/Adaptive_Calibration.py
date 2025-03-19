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