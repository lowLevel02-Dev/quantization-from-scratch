import numpy as np

from quant.core import (
    compute_qparams,
    quantize,
    dequantize,
    compute_qparams_per_channel,
    quantize_per_channel,
    dequantize_per_channel,
    mean_absolute_error,
    mean_squared_error,
    max_absolute_error,
)


# Simulated neural network weights
weights = np.array([
    [0.1, 0.2, 0.3],
    [2.0, 3.0, 4.0],
    [-0.5, 0.5, 0.2],
    [10.0, 20.0, 30.0],
])

# -----------------------------
# Per-Tensor Quantization
# -----------------------------

scale, zero_point = compute_qparams(
    weights,
    symmetric=True
)

q_tensor = quantize(weights, scale, zero_point)

reconstructed_tensor = dequantize(
    q_tensor, scale, zero_point
)

# -----------------------------
# Per-Channel Quantization
# -----------------------------

scales, zero_points = compute_qparams_per_channel(
    weights,
    symmetric=True,
    axis=0,
)

q_channel = quantize_per_channel(
    weights, scales, zero_points, axis=0
)

reconstructed_channel = dequantize_per_channel(
    q_channel, scales, zero_points, axis=0
)

# -----------------------------
# Compare Results
# -----------------------------

print("\nPER-TENSOR QUANTIZATION")
print("MAE:", mean_absolute_error(weights, reconstructed_tensor))
print("MSE:", mean_squared_error(weights, reconstructed_tensor))
print("Max Error:", max_absolute_error(weights, reconstructed_tensor))

print("\nPER-CHANNEL QUANTIZATION")
print("MAE:", mean_absolute_error(weights, reconstructed_channel))
print("MSE:", mean_squared_error(weights, reconstructed_channel))
print("Max Error:", max_absolute_error(weights, reconstructed_channel))

print("\nCOMPARISON")
print("Original Weights:\n", weights)
print("\nPer-Tensor Reconstructed:\n", reconstructed_tensor)
print("\nPer-Channel Reconstructed:\n", reconstructed_channel)