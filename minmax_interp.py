import numpy as np
import numba
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# --------------------------
# Numba-accelerated min/max downsampling
# --------------------------
@numba.njit
def minmax_downsample_indices(values, n_bins):
    """
    Compute selected indices for downsampling using min and max per bin.
    """
    N = values.shape[0]
    if n_bins >= N:
        out = np.empty(N, dtype=np.int64)
        for i in range(N):
            out[i] = i
        return out

    max_possible = 2 * n_bins + 2
    selected = np.empty(max_possible, dtype=np.int64)
    count = 0

    # Always include the first index.
    selected[count] = 0
    count += 1

    for i in range(n_bins):
        start = (i * N) // n_bins
        end = ((i + 1) * N) // n_bins
        if start >= end:
            continue
        min_idx = start
        max_idx = start
        min_val = values[start]
        max_val = values[start]
        for j in range(start, end):
            val = values[j]
            if val < min_val:
                min_val = val
                min_idx = j
            if val > max_val:
                max_val = val
                max_idx = j
        selected[count] = min_idx
        count += 1
        selected[count] = max_idx
        count += 1

    # Always include the last index.
    selected[count] = N - 1
    count += 1

    # Trim to actual count.
    out = selected[:count]
    out = np.sort(out)

    # Remove duplicates manually.
    unique_count = 1
    for i in range(1, out.shape[0]):
        if out[i] != out[i - 1]:
            out[unique_count] = out[i]
            unique_count += 1
    return out[:unique_count]


def minmax_downsample(time_arr, values, n_bins):
    indices = minmax_downsample_indices(values, n_bins)
    return time_arr[indices], values[indices]


import numpy as np
import numba
import time


@numba.njit(parallel=True)
def minmax_downsample_indices_parallel(values, n_bins):
    """
    Compute selected indices for downsampling using min and max per bin.
    This version attempts parallel processing over bins.
    """
    N = values.shape[0]
    # Pre-calculate bin boundaries.
    bin_starts = np.empty(n_bins, dtype=np.int64)
    bin_ends = np.empty(n_bins, dtype=np.int64)
    for i in numba.prange(n_bins):
        bin_starts[i] = (i * N) // n_bins
        bin_ends[i] = ((i + 1) * N) // n_bins

    # Allocate maximum possible space for indices.
    max_possible = 2 * n_bins + 2
    selected = np.empty(max_possible, dtype=np.int64)
    count = 0

    # Always include the first index.
    selected[count] = 0
    count += 1

    # For parallel reduction, store results in temporary arrays.
    temp_min = np.empty(n_bins, dtype=np.int64)
    temp_max = np.empty(n_bins, dtype=np.int64)

    for i in numba.prange(n_bins):
        start = bin_starts[i]
        end = bin_ends[i]
        if start >= end:
            temp_min[i] = -1
            temp_max[i] = -1
        else:
            min_idx = start
            max_idx = start
            min_val = values[start]
            max_val = values[start]
            for j in range(start, end):
                val = values[j]
                if val < min_val:
                    min_val = val
                    min_idx = j
                if val > max_val:
                    max_val = val
                    max_idx = j
            temp_min[i] = min_idx
            temp_max[i] = max_idx

    # Collect the results from temporary arrays.
    for i in range(n_bins):
        if temp_min[i] != -1:
            selected[count] = temp_min[i]
            count += 1
            selected[count] = temp_max[i]
            count += 1

    # Always include the last index.
    selected[count] = N - 1
    count += 1

    # Trim and sort the indices.
    out = selected[:count]
    # Simple insertion sort (since the array is nearly sorted) to avoid Python overhead.
    for i in range(1, out.shape[0]):
        key = out[i]
        j = i - 1
        while j >= 0 and out[j] > key:
            out[j + 1] = out[j]
            j -= 1
        out[j + 1] = key

    # Remove duplicates in place.
    unique_count = 1
    for i in range(1, out.shape[0]):
        if out[i] != out[i - 1]:
            out[unique_count] = out[i]
            unique_count += 1
    return out[:unique_count]


def minmax_downsample_parallel(time_arr, values, n_bins):
    indices = minmax_downsample_indices_parallel(values, n_bins)
    return time_arr[indices], values[indices]


# --------------------------
# Interpolation Downsampling
# --------------------------
def downsample_interpolate(time_arr, values, n_bins):
    new_time = np.linspace(time_arr[0], time_arr[-1], n_bins)
    new_values = np.interp(new_time, time_arr, values)
    return new_time, new_values


# --------------------------
# Create a test signal
# --------------------------
N = 1_000_000
time_arr = np.linspace(0, 10, N)
# A noisy sine wave that includes negative values
values = np.sin(time_arr) - 0.5 + np.random.randn(N) * 0.1

n_bins = 1000

# Warm up the Numba function
minmax_downsample(time_arr, values, n_bins)

# Benchmark and compute min/max downsampling.
start = time.time()
t_minmax, v_minmax = minmax_downsample(time_arr, values, n_bins)
minmax_time = (time.time() - start) * 1000

# Benchmark and compute interpolation downsampling.
start = time.time()
t_interp, v_interp = downsample_interpolate(time_arr, values, n_bins)
interp_time = (time.time() - start) * 1000

# Test the parallel version with the same signal.
N = 1_000_000
time_arr = np.linspace(0, 10, N)
values = np.sin(time_arr) - 0.5 + np.random.randn(N) * 0.1

n_bins = 1000

# Warm up the parallel function.
minmax_downsample_parallel(time_arr, values, n_bins)

start = time.time()
t_minmax_par, v_minmax_par = minmax_downsample_parallel(time_arr, values, n_bins)
parallel_time = (time.time() - start) * 1000


print(f"Min/Max Downsampling time: {minmax_time:.2f} ms")
print(f"Parallel Min/Max Downsampling time: {parallel_time:.2f} ms")
print(f"Interpolation Downsampling time: {interp_time:.2f} ms")

# --------------------------
# Plot with Plotly
# --------------------------
# Create a Plotly figure with two subplots: one for each method.
fig = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=("Min/Max Downsampling", "Interpolation Downsampling"),
    shared_xaxes=True,
)

# Original signal (a light background trace) for context.
orig_trace = go.Scattergl(
    x=time_arr,
    y=values,
    mode="lines",
    line=dict(color="lightgray"),
    name="Original Signal",
)

# Min/Max downsampled trace
minmax_trace = go.Scattergl(
    x=t_minmax,
    y=v_minmax,
    mode="lines+markers",
    marker=dict(color="red", size=3),
    line=dict(color="red"),
    name="Min/Max Downsampled",
)

# Interpolated trace
interp_trace = go.Scattergl(
    x=t_interp,
    y=v_interp,
    mode="lines+markers",
    marker=dict(color="blue", size=3),
    line=dict(color="blue"),
    name="Interpolated Downsampled",
)

# Add traces to the subplots.
fig.add_trace(orig_trace, row=1, col=1)
fig.add_trace(minmax_trace, row=1, col=1)

fig.add_trace(orig_trace, row=2, col=1)
fig.add_trace(interp_trace, row=2, col=1)

# Update layout for clarity.
fig.update_layout(
    height=800,
    width=900,
    title_text="Comparison of Downsampling Methods",
    xaxis_title="Time",
    yaxis_title="Signal Value",
)

fig.update_xaxes(title_text="Time")
fig.update_yaxes(title_text="Signal Value")

# Show the interactive plot.
fig.show()
