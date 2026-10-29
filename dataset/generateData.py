import pandas as pd
import numpy as np

lines = int(input("How many lines of data do you want to generate? "))

dx = np.random.uniform(-1000, 1000, lines)
dy = np.random.uniform(-1000, 1000, lines)

# Create DataFrame directly
df = pd.DataFrame({'dx': dx, 'dy': dy})

# Pandas calculates and creates the 'ang' column instantly
df['ang'] = np.arctan2(df['dy'], df['dx'])

# Save to disk
df.to_csv("pointer_data.csv", index=False)
print(df.head())