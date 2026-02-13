
import os

import pandas as pd
from dotenv import load_dotenv
import gridstatus

# Load the .env file
load_dotenv()

# Verify the environment variable is picked up
api_key = os.getenv("PJM_API_KEY")

if not api_key:
    print("Waiting for PJM email approval. Using public data mode...")
    # Some gridstatus methods work without a key for limited public data
    iso = gridstatus.PJM()
else:
    iso = gridstatus.PJM(api_key=api_key)


# Calculate yesterday's date programmatically
yesterday = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

# Now pass the formatted string
# Fetch Day-Ahead Hourly LMPs for yesterday
df = iso.get_lmp(
    date=yesterday,
    market="DAY_AHEAD_HOURLY",
    location_type="ZONE"
)

print(df.head())