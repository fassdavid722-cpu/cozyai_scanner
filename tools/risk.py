def calculate_risk(balance):
    if balance < 10:
        return 0.20
    elif balance < 100:
        return 0.05
    elif balance < 1000:
        return 0.03
    return 0.02
