
import numpy as np

from quant.core import (
    compute_qparams,
    quantize,
    dequantize,
    mean_absolute_error,
    mean_squared_error,
    max_absolute_error
)

# Original floating-point data
x = np.array([-1.0, 0.0, 3.0], dtype=np.float32)

# Step 1: Calculate parameters
scale, zero_point = compute_qparams(
    x,
    symmetric=False
)

# Step 2: Quantize
q = quantize(x, scale, zero_point)

# Step 3: Dequantize
x_hat = dequantize(q, scale, zero_point)

# Step 4: Calculate errors
mae = mean_absolute_error(x, x_hat)
mse = mean_squared_error(x, x_hat)
max_error = max_absolute_error(x, x_hat)

# Display results
print("Original:", x)
print("Scale:", scale)
print("Zero Point:", zero_point)
print("Quantized:", q)
print("Dequantized:", x_hat)

print("\n--- Error Analysis ---")
print("MAE:", mae)
print("MSE:", mse)
print("Maximum Error:", max_error)
