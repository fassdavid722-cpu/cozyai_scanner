def calculate_risk(balance, score):
    """
    Adaptive risk model:
    - small account = survival mode
    - high confidence = more allocation
    """

    base_risk = 0.02

    if balance < 10:
        base_risk = 0.05
    elif balance < 100:
        base_risk = 0.03
    else:
        base_risk = 0.01

    return balance * base_risk * score
