# INT8 Quantization from Scratch

An educational quantization library written in Python and NumPy.

This project shows how floating-point values can be represented as
signed 8-bit integers, how the quantization parameters (scale and zero
point) are calculated, and how per-channel quantization can reduce
reconstruction error compared with per-tensor quantization.

> **Scope:** This repository implements numerical quantization
> operations and experiments in NumPy. It is **not** a production
> inference runtime and does not quantize or execute a PyTorch model.

## Features

-   Symmetric and asymmetric quantization parameter calculation
-   Per-tensor quantization and dequantization
-   Per-channel quantization and dequantization along a chosen axis
-   INT8 rounding and clipping
-   Input validation (empty arrays, non-finite values, invalid ranges,
    incompatible shapes)
-   Reconstruction metrics: mean absolute error (MAE), mean squared
    error (MSE), maximum absolute error
-   Automated tests with `pytest`
-   An experiment comparing per-tensor and per-channel reconstruction
    error

## Technology stack

-   **Python** --- core implementation
-   **NumPy** --- numerical computation and array operations
-   **pytest** --- automated testing
-   **Git** --- version control

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

### Notation

  -----------------------------------------------------------------------
  Symbol                              Meaning
  ----------------------------------- -----------------------------------
  `x`                                 Original floating-point value

  `q`                                 Quantized integer value

  `x_hat`                             Value reconstructed after
                                      dequantization

  `s`                                 Scale: the size of one integer
                                      step, in floating-point units

  `z`                                 Zero point: the integer that
                                      represents real `0.0`

  `q_min`, `q_max`                    Integer range (default for signed
                                      INT8: `-128`, `127`)
  -----------------------------------------------------------------------

### Quantization and dequantization

``` text
Quantize:     q     = clip( round(x / s + z), q_min, q_max )
Dequantize:   x_hat = s * (q - z)
```

The reconstruction is approximate because the signed INT8 representation
provides up to 256 integer levels. The symmetric implementation uses 255
levels (`[-127, 127]`) to maintain equal positive and negative ranges.
When no clipping occurs, the rounding error for a value is at most
`s / 2`.

### Symmetric quantization

The zero point is fixed at `0`, and the scale comes from the largest
absolute value in the input:

``` text
s = max(|x|) / 127
z = 0
```

The scale uses `127` (not `128`) so that the positive and negative
ranges are equal, `[-127, 127]`. The value `-128` is left unused.

Best suited to data centered on zero, such as many neural-network weight
tensors.

### Asymmetric quantization

The scale and zero point are calculated so that the observed minimum and
maximum map onto the configured integer range:

``` text
x_min = min(x, 0)          (zero is always included in the range)
x_max = max(x, 0)

s = (x_max - x_min) / (q_max - q_min)
z = clip( round(q_min - x_min / s), q_min, q_max )
```

Because `z` is an integer, real `0.0` is represented exactly by the
affine mapping. This is useful for data that is not centered on zero,
including non-negative activations.

For an all-zero input, the implementation uses `s = 1.0` and `z = 0` to
avoid division by zero.

### Per-tensor vs. per-channel quantization

  -----------------------------------------------------------------------
                          Per-tensor              Per-channel
  ----------------------- ----------------------- -----------------------
  Parameters              One `(s, z)` pair for   One `(s, z)` pair per
                          the whole tensor        slice along `axis`

  Simplicity              Simplest approach       Requires per-channel
                                                  parameter handling

  Accuracy                A large-valued channel  Each channel gets a
                          can force a coarse      scale suited to its own
                          scale on other channels range
  -----------------------------------------------------------------------

For example, if the rows of a weight matrix have very different
magnitudes, a single scale is dominated by the largest row, and the
small rows lose precision. One scale per row can reduce this effect.

The `axis` argument selects the dimension along which channel parameters
are calculated and applied.

## Quantization workflow

The library follows this high-level numerical workflow:

``` mermaid
flowchart TD
    A[Original floating-point tensor] --> B[Calculate scale and zero point]
    B --> C[Quantize and clip]
    C --> D[INT8 tensor]
    D --> E[Dequantize]
    E --> F[Reconstructed floating-point tensor]
    A --> G[Compare original and reconstructed values]
    F --> G
    G --> H[MAE, MSE, maximum absolute error]
```

## Installation

Create and activate a virtual environment.

**Windows (PowerShell)**

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

``` bash
python -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

``` bash
python -m pip install -r requirements.txt
```

The project dependencies are `numpy` and `pytest`.

## Run the tests

From the repository root:

``` bash
python -m pytest tests/ -v
```

The development test run reported **28 passing tests**. Run the command
above to verify the current state of your checkout.

## Run the comparison experiment

From the repository root:

``` bash
python -m experiments.compare_quantization
```

The experiment quantizes a small matrix using per-tensor and per-channel
**symmetric** quantization, dequantizes the results, and prints MAE,
MSE, and maximum absolute error.

### Recorded results

The matrix was constructed so that each row has a different magnitude
(axis `0` is the channel axis):

``` python
[
    [  0.1,  0.2,  0.3],
    [  2.0,  3.0,  4.0],
    [ -0.5,  0.5,  0.2],
    [ 10.0, 20.0, 30.0],
]
```

  Metric                     Per-tensor   Per-channel
  ------------------------ ------------ -------------
  MAE                        0.05380577    0.01528871
  MSE                        0.00400273    0.00105932
  Maximum absolute error     0.11023622    0.07874016

Per-channel quantization reduced MAE by about **71.6%** and MSE by about
**73.5%** on this input.

**Interpretation:** The per-tensor scale is set by the value `30.0`, so
it is about `0.236` for every row. That step is large relative to many
values in the first three rows, which is why they reconstruct less
accurately. Per-channel quantization gives those rows finer scales.

**Limitation:** This matrix is deliberately constructed with
substantially different row magnitudes to make the effect easy to
observe. The result does not guarantee the same improvement on real
model weights, where the gap may be smaller or may vary by model and
tensor.

## API overview

The implementation is in `quant/core.py`.

  ------------------------------------------------------------------------------------------------------
  Function                                                           Purpose
  ------------------------------------------------------------------ -----------------------------------
  `compute_qparams(x, symmetric, qmin, qmax)`                        Calculate the scale and zero point
                                                                     for a tensor

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

Expected output:

``` text
Scale: 0.007874015748031496
Zero point: 0
Quantized: [-127  -32   64  127]
Reconstructed: [-1.         -0.2519685   0.50393701  1.        ]
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
    [  0.1,  0.2,  0.3],
    [  2.0,  3.0,  4.0],
    [ -0.5,  0.5,  0.2],
    [ 10.0, 20.0, 30.0],
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
    within `[-128, 127]`; custom ranges outside this interval are not
    representable by the returned `int8` dtype.
-   Per-channel parameter arrays must contain one scale and one zero
    point for each position along the selected axis.
-   Use the same axis and corresponding parameters for both quantization
    and dequantization.
-   Reconstruction is approximate because of rounding and clipping.
-   This project focuses on numerical fundamentals. It does **not**
    include model graph conversion, calibration pipelines, operator
    fusion, hardware-specific kernels, or inference benchmarking.
-   The implementation is educational and has not been validated as a
    production quantization framework.

## Engineering lessons

Several implementation issues were identified and corrected during
development:

-   **Missing required argument:** `compute_qparams` requires an
    explicit `symmetric` argument. The experiment call was updated to
    pass the intended quantization mode.
-   **Empty per-channel input:** Iterating over an empty channel
    dimension could return empty parameter arrays without raising an
    error. An explicit empty-input validation check was added.
-   **Dequantization variable mismatch:** Reconstruction must operate on
    the quantized input `q`. The implementation was corrected to use the
    quantized tensor consistently.
-   **Floating-point test expectations:** A test initially expected an
    exact decimal reconstruction. Its expected value was updated to
    account for the rounded zero point and quantization approximation.
-   **MSE test expectation:** The expected MSE was recalculated from the
    element-wise squared errors.

These are documented development lessons, not known unresolved defects.

## Possible future extensions

-   [ ] Apply per-channel quantization to real PyTorch `Linear` layer
    weights
-   [ ] Add activation quantization and calibration
-   [ ] Compare symmetric and asymmetric quantization over varied inputs
-   [ ] Expand tests across tensor shapes, axes, and edge cases
-   [ ] Benchmark model size, inference latency, and accuracy
-   [ ] Explore hardware-oriented INT8 inference


