import os
from dotenv import load_dotenv

load_dotenv()

BITGET_API_KEY = os.getenv("BITGET_API_KEY")
BITGET_SECRET = os.getenv("BITGET_SECRET")
BITGET_PASSWORD = os.getenv("BITGET_PASSWORD")

TIMEFRAME = "1m"
SCAN_LIMIT = 30

# Safety controls
MAX_TRADES_PER_CYCLE = 1
MIN_SCORE = 0.72
COOLDOWN_SECONDS = 60

DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
