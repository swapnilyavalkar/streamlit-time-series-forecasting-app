import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate date range for 5 years (from today's date)
date_range = pd.date_range(end=datetime.today(), periods=5*365, freq='D')

# Create time series data for different columns
data = {
    'date': date_range,
    'value_A': np.random.randn(len(date_range)) * 100 + 1000,  # Random data for value_A
    'value_B': np.random.randn(len(date_range)) * 50 + 500,    # Random data for value_B
    'value_C': np.random.randn(len(date_range)) * 200 + 2000   # Random data for value_C
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save as Excel, CSV, and text file
df.to_csv('random-data-files\\time_series_data.csv', index=False)
df.to_excel('random-data-files\\time_series_data.xlsx', index=False)
df.to_csv('random-data-files\\time_series_data.txt', index=False, sep='\t')

import ace_tools as tools; tools.display_dataframe_to_user(name="5 Year Time Series Data", dataframe=df)