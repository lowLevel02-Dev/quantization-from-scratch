import numpy as np
from quant.core import (
    compute_qparams_per_channel,
    quantize_per_channel,
    dequantize_per_channel,
)


def test_per_channel_quantize_shape():
    x = np.array([
        [0.1, 0.2, 0.3],
        [2.0, 3.0, 4.0],
        [-0.5, 0.5, 0.2],
        [10.0, 20.0, 30.0]
    ])

    scales, zero_points = compute_qparams_per_channel(x)

    q = quantize_per_channel(x, scales, zero_points)

    assert q.shape == x.shape
    assert q.dtype == np.int8


def test_per_channel_dequantize_shape():
    x = np.array([
        [0.1, 0.2, 0.3],
        [2.0, 3.0, 4.0],
        [-0.5, 0.5, 0.2],
        [10.0, 20.0, 30.0]
    ])

    scales, zero_points = compute_qparams_per_channel(x)

    q = quantize_per_channel(x, scales, zero_points)

    reconstructed = dequantize_per_channel(
        q, scales, zero_points
    )

    assert reconstructed.shape == x.shape


def test_per_channel_quantization_error():
    x = np.array([
        [0.1, 0.2, 0.3],
        [2.0, 3.0, 4.0],
        [-0.5, 0.5, 0.2],
        [10.0, 20.0, 30.0]
    ])

    scales, zero_points = compute_qparams_per_channel(x)

    q = quantize_per_channel(x, scales, zero_points)

    reconstructed = dequantize_per_channel(
        q, scales, zero_points
    )

    error = np.mean(np.abs(x - reconstructed))

    assert error < 0.1


def test_per_channel_axis_1():
    x = np.array([
        [0.1, 2.0, 0.5],
        [0.2, 3.0, 0.2],
        [0.3, 4.0, 0.1]
    ])

    scales, zero_points = compute_qparams_per_channel(
        x, axis=1
    )

    q = quantize_per_channel(
        x, scales, zero_points, axis=1
    )

    reconstructed = dequantize_per_channel(
        q, scales, zero_points, axis=1
    )

    assert q.shape == x.shape
    assert reconstructed.shape == x.shape