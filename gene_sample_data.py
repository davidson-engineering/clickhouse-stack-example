import random
from clickhouse_driver import Client
import math
import time
from datetime import datetime, timezone


client = Client(
    host="localhost",
    port=9000,
    user="default",  # Explicitly set user
    password="your_secure_password",  # Use an empty password if that's the default
)
# Drop the table if it already exists (for testing purposes)
client.execute("DROP TABLE IF EXISTS sample_data")

# Create a table to store the wave data with an additional wave_id column.
client.execute(
    """
CREATE TABLE sample_data (
    wave_id UInt16,      -- identifies the wave
    rel_time UInt64,     -- relative time in microseconds for each wave
    value Float64
) ENGINE = MergeTree()
PARTITION BY wave_id
ORDER BY (wave_id, rel_time)
"""
)

# Parameters for the sample data:
num_waves = 100  # Generate 100 waves
num_points = 500000  # 50ms with 1µs resolution → 50,000 points per wave
T = 0.05  # Total duration in seconds (50ms)


def get_amplitude():
    return random.uniform(0.2, 2)


data = []

for wave in range(num_waves):
    # Define frequency sweep for each wave.
    f_start = 1 + wave * 5  # Starting frequency increases with wave id
    f_end = f_start + 20  # Ending frequency is 2000Hz higher than start
    amp = get_amplitude()

    for i in range(num_points):
        # Use the loop counter as relative time in microseconds.
        rel_time = i  # 0, 1, 2, ... 49999 (µs)
        # Convert microseconds to seconds for the sine calculation.
        t = i * 1e-6  # current time in seconds

        # Compute instantaneous phase:
        # φ(t) = f_start*t + ((f_end - f_start) / (2*T)) * t^2
        phi = f_start * t + ((f_end - f_start) / (2 * T)) * (t**2)
        # Calculate the sine value with frequency modulation.
        value = amp * math.sin(2 * math.pi * phi)

        data.append((wave, rel_time, value))

# Insert the generated data into ClickHouse.
client.execute("INSERT INTO sample_data (wave_id, rel_time, value) VALUES", data)

print(f"Inserted {len(data)} data points for {num_waves} waves into ClickHouse.")
