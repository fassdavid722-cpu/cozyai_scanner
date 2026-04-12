from brain.agent import think

def run():
    balance = 3.0

    decision = think(balance)

    print("AI decision:")
    print(decision)

if __name__ == "__main__":
    run()
