# INT8 Quantization from Scratch

An educational quantization library implemented in Python and NumPy.
This project explores how floating-point values can be represented with
signed 8-bit integers, how quantization parameters are calculated, and
how per-channel quantization can reduce reconstruction error compared
with per-tensor quantization.

> **Scope:** This repository implements numerical quantization
> operations and experiments in NumPy. It is not a production inference
> runtime and does not currently quantize or execute a PyTorch model.

## Features

-   Symmetric and asymmetric quantization parameter calculation
-   Per-tensor quantization and dequantization
-   Per-channel quantization and dequantization along a selected axis
-   INT8 rounding and clipping
-   Input validation for empty arrays, non-finite values, invalid
    ranges, and incompatible shapes
-   Reconstruction metrics: mean absolute error (MAE), mean squared
    error (MSE), and maximum absolute error
-   Automated tests with `pytest`
-   An experiment comparing per-tensor and per-channel reconstruction
    error

## Repository structure

``` text
quantization-from-scratch/
├── quant/
│   ├── __init__.py
│   └── core.py
├── tests/
│   ├── test_qparams.py
│   └── test_per_channel.py
├── experiments/
│   └── compare_quantization.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Concepts

### Quantization

Quantization maps floating-point values to a discrete integer
representation. The usual signed INT8 range is:

\[ q\_{`\min`{=tex}}=-128,`\qquad `{=tex}q\_{`\max`{=tex}}=127 \]

The quantized value is calculated using a scale (s) and zero point (z):

\[ q = `\operatorname{clip}`{=tex}`\left`{=tex}(
`\operatorname{round}`{=tex}`\left`{=tex}(`\frac{x}{s}`{=tex}+z`\right`{=tex}),
q\_{`\min`{=tex}},q\_{`\max`{=tex}} `\right`{=tex}) \]

Dequantization reconstructs an approximation of the original value:

\[ `\hat{x}`{=tex}=s(q-z) \]

The reconstruction is generally approximate because the integer
representation has a finite number of representable levels.

### Symmetric quantization

Symmetric quantization uses a zero point of zero. The scale is
calculated from the largest absolute input magnitude. With the default
signed INT8 range, this implementation uses the symmetric range
(\[-127,127\]) to calculate the scale:

\[ s=`\frac{\max(|x_{\min}|,|x_{\max}|)}{127}`{=tex} \]

### Asymmetric quantization

Asymmetric quantization uses the configured integer range and calculates
a zero point to represent real zero:

\[ s=`\frac{x_{\max}-x_{\min}}{q_{\max}-q_{\min}}`{=tex} \]

\[ z=`\operatorname{clip}`{=tex}`\left`{=tex}(
`\operatorname{round}`{=tex}`\left`{=tex}(q\_{`\min`{=tex}}-`\frac{x_{\min}}{s}`{=tex}`\right`{=tex}),
q\_{`\min`{=tex}},q\_{`\max`{=tex}} `\right`{=tex}) \]

The implementation includes zero in the observed real-valued range. For
an all-zero input, it uses scale `1.0` and zero point `0` to avoid
division by zero.

### Per-tensor vs. per-channel quantization

**Per-tensor quantization** calculates one scale and zero point for the
entire tensor.

**Per-channel quantization** calculates separate parameters for each
slice along a selected axis. This can help when channels have
substantially different value ranges. For example, if rows of a weight
matrix have different magnitudes, one scale per row can preserve
smaller-valued rows more accurately than one scale for the entire
matrix.

The `axis` argument selects the dimension along which channel parameters
are calculated and applied.

## Installation

Create and activate a virtual environment.

**Windows PowerShell**

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

``` bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

``` bash
python -m pip install -r requirements.txt
```

If you are setting up without a requirements file, the core dependencies
are NumPy and pytest:

``` bash
python -m pip install numpy pytest
```

## Run the tests

From the repository root:

``` bash
python -m pytest tests/ -v
```

The development test run reported **28 passing tests**. Run the command
above in your environment to verify the current repository state.

## Run the comparison experiment

From the repository root:

``` bash
python -m experiments.compare_quantization
```

The experiment quantizes a sample matrix of simulated neural-network
weights using both approaches, dequantizes the results, and prints MAE,
MSE, and maximum absolute error.

### Recorded experiment

The sample matrix used during development was:

``` python
[
    [0.1,  0.2,  0.3],
    [2.0,  3.0,  4.0],
    [-0.5, 0.5,  0.2],
    [10.0, 20.0, 30.0],
]
```

Observed results:

  Metric                     Per-tensor   Per-channel
  ------------------------ ------------ -------------
  MAE                        0.05380577    0.01528871
  MSE                        0.00400273    0.00105932
  Maximum absolute error     0.11023622    0.07874016

For this input, per-channel quantization reduced MAE by approximately
**71.6%** and MSE by approximately **73.5%**. These results apply to
this example; they do not guarantee that per-channel quantization will
always produce lower error for every tensor or model.

## API overview

The main implementation is in `quant/core.py`.

  ------------------------------------------------------------------------------------------------------
  Function                                                           Purpose
  ------------------------------------------------------------------ -----------------------------------
  `compute_qparams(x, symmetric, qmin, qmax)`                        Calculate a tensor's scale and zero
                                                                     point

  `quantize(x, scale, zero_point, qmin, qmax)`                       Quantize values to INT8

  `dequantize(q, scale, zero_point)`                                 Reconstruct approximate
                                                                     floating-point values

  `compute_qparams_per_channel(x, symmetric, axis, qmin, qmax)`      Calculate parameters for each
                                                                     channel

  `quantize_per_channel(x, scales, zero_points, axis, qmin, qmax)`   Quantize with channel-specific
                                                                     parameters

  `dequantize_per_channel(q, scales, zero_points, axis)`             Dequantize with channel-specific
                                                                     parameters

  `mean_absolute_error(original, reconstructed)`                     Calculate MAE

  `mean_squared_error(original, reconstructed)`                      Calculate MSE

  `max_absolute_error(original, reconstructed)`                      Calculate maximum absolute error
  ------------------------------------------------------------------------------------------------------

### Basic usage

``` python
import numpy as np

from quant.core import compute_qparams, quantize, dequantize

values = np.array([-1.0, -0.25, 0.5, 1.0])

scale, zero_point = compute_qparams(values, symmetric=True)
quantized = quantize(values, scale, zero_point)
reconstructed = dequantize(quantized, scale, zero_point)

print("Scale:", scale)
print("Zero point:", zero_point)
print("Quantized:", quantized)
print("Reconstructed:", reconstructed)
```

### Per-channel usage

``` python
import numpy as np

from quant.core import (
    compute_qparams_per_channel,
    quantize_per_channel,
    dequantize_per_channel,
)

weights = np.array([
    [0.1, 0.2, 0.3],
    [2.0, 3.0, 4.0],
    [-0.5, 0.5, 0.2],
    [10.0, 20.0, 30.0],
])

scales, zero_points = compute_qparams_per_channel(
    weights, symmetric=True, axis=0
)

quantized = quantize_per_channel(
    weights, scales, zero_points, axis=0
)

reconstructed = dequantize_per_channel(
    quantized, scales, zero_points, axis=0
)
```

## Implementation notes and limitations

-   Quantized outputs are NumPy `int8` arrays. Keep `qmin` and `qmax`
    within `[-128, 127]`.
-   Per-channel parameter arrays must contain one scale and zero point
    for each position along the selected axis.
-   Use the same axis and corresponding parameters for quantization and
    dequantization.
-   Floating-point reconstruction is approximate because of rounding and
    clipping.
-   This project focuses on numerical fundamentals. It does not
    currently include model graph conversion, calibration pipelines,
    operator fusion, hardware-specific kernels, or inference
    benchmarking.
-   The implementation is educational and is not a validated production
    quantization framework.

## Development notes

Several issues were identified and corrected while building the project:

-   **Missing required argument:** `compute_qparams` requires an
    explicit `symmetric` argument. The experiment call was corrected to
    pass the intended quantization mode.
-   **Empty per-channel input:** Iterating over an empty channel
    dimension could return empty parameter arrays without raising an
    error. An explicit empty-input check was added.
-   **Dequantization variable mismatch:** Reconstruction must use the
    quantized input (q). The implementation was corrected to use the
    appropriate variable.
-   **Floating-point test expectations:** A test initially expected an
    exact decimal reconstruction. Its expected value was updated to
    reflect the rounded zero point and quantization approximation.
-   **MSE test expectation:** The expected MSE was recalculated from the
    element-wise squared errors.

These notes document the development and testing process; they are not a
list of known unresolved defects.

## Possible future extensions

-   Apply per-channel quantization to actual PyTorch `Linear` layer
    weights
-   Add activation quantization and calibration
-   Compare symmetric and asymmetric quantization over varied inputs
-   Expand tests across tensor shapes, axes, and edge cases
-   Benchmark model size, inference latency, and accuracy
-   Explore hardware-oriented INT8 inference

