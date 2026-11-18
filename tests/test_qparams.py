import numpy as np 
import pytest  

from quant.core import compute_qparams,quantize,dequantize,mean_absolute_error,mean_squared_error,max_absolute_error,compute_qparams_per_channel

def test_asymetric_basic():
    x = np.array([-1.0,0.0,3.0])

    scale,zero_point = compute_qparams(x , symmetric= False)
    expected_scale = 4.0 / 255.0

    assert np.isclose(scale , expected_scale)
    assert zero_point == -64
def test_symmetric_basic():
    x = np.array([-5.0 , 2.0])
    scale,zero_point = compute_qparams(x , symmetric=True)
    expected_scale = 5.0/127.0
    assert np.isclose(scale, expected_scale)
    assert zero_point == 0
def test_all_positive_asymmetric():

    x = np.array([1.0, 2.0, 3.0])

    scale, zero_point = compute_qparams(
        x,
        symmetric=False
    )

    assert zero_point == -128
def test_all_zeros():

    x = np.array([0.0, 0.0, 0.0])

    scale, zero_point = compute_qparams(
        x,
        symmetric=False
    )

    assert scale == 1.0
    assert zero_point == 0
def test_nan_rejected():

    x = np.array([1.0, np.nan, 3.0])

    with pytest.raises(ValueError):
        compute_qparams(x, symmetric=False)
def test_inf_rejected():

    x = np.array([1.0, np.inf, 3.0])

    with pytest.raises(ValueError):
        compute_qparams(x, symmetric=False)


def test_empty_array_rejected():

    x = np.array([])

    with pytest.raises(ValueError):
        compute_qparams(x, symmetric=False)

def test_quantize_asymmetric():

    x = np.array([-1.0, 0.0, 3.0])

    scale, zero_point = compute_qparams(
        x,
        symmetric=False
    )

    result = quantize(x, scale, zero_point)

    expected = np.array([-128, -64, 127], dtype=np.int8)

    assert result.dtype == np.int8
    assert np.array_equal(result, expected)


def test_quantize_symmetric():

    x = np.array([-5.0, 2.0])

    scale, zero_point = compute_qparams(
        x,
        symmetric=True
    )

    result = quantize(x, scale, zero_point)

    expected = np.array([-127, 51], dtype=np.int8)

    assert np.array_equal(result, expected)


def test_quantize_clipping():

    x = np.array([-1000.0, 1000.0])

    result = quantize(
        x,
        scale=0.01,
        zero_point=0
    )

    expected = np.array([-128, 127], dtype=np.int8)

    assert np.array_equal(result, expected)


def test_quantize_zero():

    x = np.array([0.0, 0.0, 0.0])

    scale, zero_point = compute_qparams(
        x,
        symmetric=False
    )

    result = quantize(x, scale, zero_point)

    expected = np.array([0, 0, 0], dtype=np.int8)

    assert np.array_equal(result, expected)


def test_quantize_invalid_scale():

    x = np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        quantize(x, scale=0.0, zero_point=0)
def test_dequantize_basic():

    q = np.array([-128, -64, 127], dtype=np.int8)

    scale = 4.0 / 255.0
    zero_point = -64

    result = dequantize(q, scale, zero_point)

    expected = np.array([
        -1.00392157,
        0.0,
        2.99607843
    ])
    expected = scale * (
    q.astype(np.float64) - zero_point
    )

    assert np.allclose(result, expected)


def test_dequantize_zero():

    q = np.array([-64, -64, -64], dtype=np.int8)

    result = dequantize(
        q,
        scale=0.05,
        zero_point=-64
    )

    assert np.allclose(result, [0.0, 0.0, 0.0])


def test_mae():

    original = np.array([-1.0, 0.0, 3.0])

    reconstructed = np.array([
        -1.0039215686,
        0.0,
        2.9960784314
    ])

    result = mean_absolute_error(original, reconstructed)

    expected = 0.0026143791

    assert np.isclose(result, expected)


def test_mse():

    original = np.array([-1.0, 0.0, 3.0])

    reconstructed = np.array([
        -1.0039215686,
        0.0,
        2.9960784314
    ])

    result = mean_squared_error(original, reconstructed)

    expected = 0.0000102525

    assert np.isclose(result, expected, atol=1e-9)


def test_max_absolute_error():

    original = np.array([-1.0, 0.0, 3.0])

    reconstructed = np.array([
        -1.0039215686,
        0.0,
        2.9960784314
    ])

    result = max_absolute_error(original, reconstructed)

    expected = 0.0039215686

    assert np.isclose(result, expected)


def test_error_shape_mismatch():

    original = np.array([1.0, 2.0, 3.0])
    reconstructed = np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        mean_absolute_error(original, reconstructed)


def test_error_nan_rejected():

    original = np.array([1.0, np.nan])
    reconstructed = np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        mean_squared_error(original, reconstructed)

def test_per_channel_symmetric():

    weights = np.array([
        [0.1, 0.2, 0.3],
        [2.0, 3.0, 4.0],
        [-0.5, 0.5, 0.2],
        [10.0, 20.0, 30.0]
    ])

    scales, zero_points = compute_qparams_per_channel(
        weights,
        symmetric=True,
        axis=0
    )

    expected_scales = np.array([
        0.3 / 127,
        4.0 / 127,
        0.5 / 127,
        30.0 / 127
    ])

    assert np.allclose(scales, expected_scales)
    assert np.array_equal(
        zero_points,
        np.array([0, 0, 0, 0])
    )


def test_per_channel_asymmetric():

    weights = np.array([
        [-1.0, 0.0, 1.0],
        [0.0, 2.0, 4.0],
        [-5.0, 0.0, 5.0]
    ])

    scales, zero_points = compute_qparams_per_channel(
        weights,
        symmetric=False,
        axis=0
    )

    expected_scales = np.array([
        2.0 / 255,
        4.0 / 255,
        10.0 / 255
    ])

    assert np.allclose(scales, expected_scales)
    assert zero_points.shape == (3,)


def test_per_channel_axis_1():

    weights = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])

    scales, zero_points = compute_qparams_per_channel(
        weights,
        symmetric=True,
        axis=1
    )

    expected_scales = np.array([
        4.0 / 127,
        5.0 / 127,
        6.0 / 127
    ])

    assert np.allclose(scales, expected_scales)
    assert np.array_equal(zero_points, [0, 0, 0])


def test_per_channel_invalid_axis():

    weights = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])

    with pytest.raises(ValueError):
        compute_qparams_per_channel(
            weights,
            axis=2
        )


def test_per_channel_empty_array():

    weights = np.empty((0, 3))

    with pytest.raises(ValueError):
        compute_qparams_per_channel(weights)
