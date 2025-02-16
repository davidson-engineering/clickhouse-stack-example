import random
from clickhouse_driver import Client
import math
import time
from datetime import datetime, timezone

import numpy as np

from minmax_interp import minmax_downsample_parallel


# Generate sample data for testing
def generate_sample_data(
    num_signals, num_points, t_duration=0.05, add_noise=False, seed=None
):

    if seed is not None:
        np.random.seed(seed)

    # Parameters for amplitude and frequency
    amp_min = 0.2
    amp_max = 2.0
    f_min = 1.0
    f_max = 1000.0
    f_range = 20.0

    # Generate time array (e.g., over 10 seconds)
    time_arr = np.linspace(0, 10, num_points)

    # Random parameters for each signal (vectorized)
    f_start = np.random.uniform(f_min, f_max, size=num_signals)  # shape: (num_signals,)
    amp = np.random.uniform(amp_min, amp_max, size=num_signals)  # shape: (num_signals,)
    f_end = f_start + f_range  # shape: (num_signals,)

    # Reshape parameters for broadcasting over time points
    # f_start and f_end become (num_signals, 1) and time_arr becomes (1, num_points)
    f_start = f_start[:, np.newaxis]
    f_end = f_end[:, np.newaxis]
    amp = amp[:, np.newaxis]

    # Calculate phase for each signal and each time point:
    # phi = f_start * t + ((f_end - f_start) / (2*t_duration)) * (t**2)
    # With broadcasting, t becomes time_arr[None, :] (shape (1, num_points))
    phi = f_start * time_arr + ((f_end - f_start) / (2 * t_duration)) * (time_arr**2)

    # Compute the sine wave values. The sine function is vectorized.
    values = amp * np.sin(2 * np.pi * phi)

    if add_noise:
        noise = np.random.normal(scale=0.1, size=values.shape)
        values += noise

    return time_arr, values


def init_clickhouse_client():

    client = Client(
        host="localhost",
        port=9000,
        user="default",  # Explicitly set user
        password="your_secure_password",  # Use an empty password if that's the default
    )
    # Drop the table if it already exists (for testing purposes)
    client.execute("DROP TABLE IF EXISTS raw_data")
    client.execute("DROP TABLE IF EXISTS processed_data")

    # Create a table to store the wave data with an additional wave_id column.
    client.execute(
        """
    CREATE TABLE raw_data (
        wave_id UInt16,      -- identifies the wave
        rel_time UInt64,     -- relative time in nanoseconds for each wave
        value Float64
    ) ENGINE = MergeTree()
    PARTITION BY wave_id
    ORDER BY (wave_id, rel_time)
    """
    )

    client.execute(
        """
    CREATE TABLE processed_data (
        wave_id UInt16,      -- identifies the wave
        rel_time UInt64,     -- relative time in nanoseconds for each wave
        value Float64
    ) ENGINE = MergeTree()
    PARTITION BY wave_id
    ORDER BY (wave_id, rel_time)
    """
    )
    return client


wave_id = 0


def transform_data(time_arr, values):
    global wave_id
    if len(values.shape) == 1:
        num_waves = 1
        num_points = len(values)
        # Create an array of wave IDs for a single wave.
        wave_ids = np.repeat(wave_id, num_points)
        wave_id += 1
    else:
        num_waves, num_points = values.shape
        # Assume num_waves, num_points, time_arr, and values have already been defined.
        # Create an array of wave IDs by repeating each wave index for all its time points.
        wave_ids = np.repeat(np.arange(num_waves), num_points)

    # Create an array of relative times by tiling the time array for each wave.
    if len(time_arr.shape) == 1:
        rel_times = np.tile(time_arr, num_waves)
    else:
        # flatten the time array if it has more than one dimension
        rel_times = time_arr.flatten()

    assert len(rel_times) == len(wave_ids)
    assert len(rel_times) == len(values.flatten())

    rel_times = (rel_times * 1e9).astype(np.uint64)  # Convert to nanoseconds

    # Flatten the values array so it lines up with the other arrays.
    values_flat = values.flatten()

    # Now zip the arrays together to form an iterator of tuples.
    data = zip(wave_ids, rel_times, values_flat)
    return list(data)


if __name__ == "__main__":
    client = init_clickhouse_client()

    # Generate sample data for 1000 waves with 1000 points each
    num_waves = 10
    num_points = 5000
    time_arr, values = generate_sample_data(num_waves, num_points, add_noise=True)

    data = transform_data(time_arr, values)

    # Insert the generated data into ClickHouse.
    client.execute("INSERT INTO raw_data (wave_id, rel_time, value) VALUES", data)
    print(f"Inserted {len(data)} data points for {num_waves} waves into ClickHouse.")

    n_bins = 300

    for signal in values:
        time_arr_new, values_new = minmax_downsample_parallel(time_arr, signal, n_bins)
        data = transform_data(time_arr_new, values_new)
        client.execute(
            "INSERT INTO processed_data (wave_id, rel_time, value) VALUES", data
        )
        print(
            f"Inserted {len(data)} data points for {num_waves} waves into ClickHouse."
        )
