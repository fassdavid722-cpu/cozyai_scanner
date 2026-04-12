from telegram_bot import run_bot
from scanner_loop import run_trader_loop  # your existing loop
import threading

def start_trader():
    run_trader_loop()

def start_bot():
    run_bot()

if __name__ == "__main__":
    threading.Thread(target=start_trader).start()
    start_bot()
