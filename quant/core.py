import numpy as np 

def compute_qparams(
        x: np.ndarray,
        symmetric:bool,
        qmin: int = -128, 
        qmax: int = 127

) -> tuple[float,int]:
    if not isinstance(qmin, (int,np.integer)):
        raise TypeError("qmin must be an integer")
    if not isinstance(qmax, (int,np.integer)):
        raise TypeError("qmax must be an integer")
    if qmin >= qmax: 
        raise ValueError("qmax should be greater than qmin")
    if not isinstance(symmetric, (bool , np.bool_)):
        raise TypeError("symetric must be a boolean")

    x = np.asarray(x)

    if x.size == 0:
        raise ValueError("Input array cannot be empty")
    if not np.issubdtype(x.dtype , np.number):
        raise TypeError("Input must contain real Number")
    if np.iscomplexobj(x):
        raise TypeError("Complex numers are not supported")

    x = x.astype(np.float64)

    if not np.all(np.isfinite(x)):
        raise ValueError("Input constains NaN or Inf")

    x_min = np.min(x)
    x_max = np.max(x)

    x_min = min(x_min,0.0)
    x_max = max(x_max,0.0)

    if x_min == 0.0 and x_max == 0.0:
        return 1.0,0
    if symmetric:
        if qmin > -1 or qmax < 1:
            raise TypeError("Symmetric quantization requires zero and both positive and negative integer values")
        a = max(abs(x_min) , abs(x_max))

        max_int = min(abs(qmin) , abs(qmax))

        scale = a / max_int

        zero_point = 0
    else:
        scale = (
            (x_max - x_min) / (qmax - qmin)
        )

        zero_point = qmin - (x_min / scale)

        zero_point = np.rint(zero_point)

        zero_point = np.clip(zero_point , qmin , qmax)

        zero_point = int(zero_point)
    return float(scale) , int(zero_point)

def quantize(
    x: np.ndarray,
    scale: float,
    zero_point: int,
    qmin: int = -128,
    qmax: int = 127
) -> np.ndarray:

    if not isinstance(qmin, (int, np.integer)):
        raise TypeError("qmin must be an integer")

    if not isinstance(qmax, (int, np.integer)):
        raise TypeError("qmax must be an integer")

    if qmin >= qmax:
        raise ValueError("qmin must be smaller than qmax")

    if not np.isscalar(scale) or not np.isreal(scale):
        raise TypeError("Scale must be a real scalar")

    scale = float(scale)

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Scale must be positive and finite")

    # 3. Validate zero point
    if not isinstance(zero_point, (int, np.integer)):
        raise TypeError("Zero point must be an integer")

    if not qmin <= zero_point <= qmax:
        raise ValueError("Zero point must be within quantization range")

    x = np.asarray(x)

    if not np.issubdtype(x.dtype, np.number):
        raise TypeError("Input must contain real numbers")

    if np.iscomplexobj(x):
        raise TypeError("Complex numbers are not supported")

    x = x.astype(np.float64)

    if not np.all(np.isfinite(x)):
        raise ValueError("Input contains NaN or Inf")

    quantized = np.rint(x / scale + zero_point)

    quantized = np.clip(quantized, qmin, qmax)

    return quantized.astype(np.int8)


def dequantize(
    q: np.ndarray,
    scale: float,
    zero_point: int
) -> np.ndarray:

    # Convert quantized integers to float64
    q = np.asarray(q).astype(np.float64)

    # Apply dequantization formula
    x_hat = scale * (q - zero_point)

    return x_hat

def _validate_error_inputs(original, reconstructed):
    """Validate two arrays before calculating error metrics."""

    original = np.asarray(original)
    reconstructed = np.asarray(reconstructed)

    if original.size == 0 or reconstructed.size == 0:
        raise ValueError("Input arrays cannot be empty")

    if original.shape != reconstructed.shape:
        raise ValueError("Input arrays must have the same shape")

    if (
        not np.issubdtype(original.dtype, np.number)
        or not np.issubdtype(reconstructed.dtype, np.number)
    ):
        raise TypeError("Inputs must contain real numbers")

    if np.iscomplexobj(original) or np.iscomplexobj(reconstructed):
        raise TypeError("Complex numbers are not supported")

    original = original.astype(np.float64)
    reconstructed = reconstructed.astype(np.float64)

    if not np.all(np.isfinite(original)):
        raise ValueError("Original contains NaN or Inf")

    if not np.all(np.isfinite(reconstructed)):
        raise ValueError("Reconstructed contains NaN or Inf")

    return original, reconstructed


def mean_absolute_error(original, reconstructed):
    original, reconstructed = _validate_error_inputs(
        original, reconstructed
    )

    error = np.abs(original - reconstructed)

    return float(np.mean(error))


def mean_squared_error(original, reconstructed):
    original, reconstructed = _validate_error_inputs(
        original, reconstructed
    )

    error = original - reconstructed

    return float(np.mean(error ** 2))


def max_absolute_error(original, reconstructed):
    original, reconstructed = _validate_error_inputs(
        original, reconstructed
    )

    error = np.abs(original - reconstructed)

    return float(np.max(error))

def compute_qparams_per_channel(
    x: np.ndarray,
    symmetric: bool = True,
    axis: int = 0,
    qmin: int = -128,
    qmax: int = 127,
) -> tuple[np.ndarray, np.ndarray]:

    x = np.asarray(x)

    if x.size == 0:
        raise ValueError("Input array cannot be empty")

    if x.ndim == 0:
        raise ValueError("Input must have at least one dimension")

    if not isinstance(axis, (int, np.integer)):
        raise TypeError("Axis must be an integer")

    if not -x.ndim <= axis < x.ndim:
        raise ValueError("Invalid axis")

    axis = axis % x.ndim
    x = np.moveaxis(x, axis, 0)

    scales = []
    zero_points = []

    for channel in x:
        scale, zero_point = compute_qparams(
            channel,
            symmetric=symmetric,
            qmin=qmin,
            qmax=qmax,
        )

        scales.append(scale)
        zero_points.append(zero_point)

    return (
        np.asarray(scales, dtype=np.float64),
        np.asarray(zero_points, dtype=np.int64),
    )

def quantize_per_channel(
    x: np.ndarray,
    scales: np.ndarray,
    zero_points: np.ndarray,
    axis: int = 0,
    qmin: int = -128,
    qmax: int = 127,
) -> np.ndarray:

    x = np.asarray(x, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    zero_points = np.asarray(zero_points)

    # Validate input tensor
    if x.size == 0:
        raise ValueError("Input tensor cannot be empty")

    if not np.all(np.isfinite(x)):
        raise ValueError("Input tensor must contain finite values")

    if x.ndim == 0:
        raise ValueError("Input tensor must have at least one dimension")

    # Validate axis
    if not isinstance(axis, (int, np.integer)):
        raise TypeError("Axis must be an integer")

    if not -x.ndim <= axis < x.ndim:
        raise ValueError("Invalid axis")

    axis = axis % x.ndim

    # Validate quantization range
    if not isinstance(qmin, (int, np.integer)):
        raise TypeError("qmin must be an integer")

    if not isinstance(qmax, (int, np.integer)):
        raise TypeError("qmax must be an integer")

    if qmin >= qmax:
        raise ValueError("qmin must be less than qmax")

    # Validate scales
    if scales.ndim != 1:
        raise ValueError("Scales must be a 1D array")

    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Scales must be positive and finite")

    # Validate zero points
    if zero_points.ndim != 1:
        raise ValueError("Zero points must be a 1D array")

    if not np.issubdtype(zero_points.dtype, np.integer):
        raise TypeError("Zero points must be integers")

    if np.any(zero_points < qmin) or np.any(zero_points > qmax):
        raise ValueError("Zero points must be within quantization range")

    # Check parameter count
    if len(scales) != x.shape[axis] or len(zero_points) != x.shape[axis]:
        raise ValueError("Number of scales and zero points must match the selected axis")

    # Reshape parameters for broadcasting
    broadcast_shape = [1] * x.ndim
    broadcast_shape[axis] = len(scales)

    scales = scales.reshape(broadcast_shape)
    zero_points = zero_points.reshape(broadcast_shape)

    # Quantize
    q = np.rint(x / scales + zero_points)
    q = np.clip(q, qmin, qmax)

    return q.astype(np.int8)

def dequantize_per_channel(
    q: np.ndarray,
    scales: np.ndarray,
    zero_points: np.ndarray,
    axis: int = 0,
) -> np.ndarray:

    q = np.asarray(q)
    scales = np.asarray(scales, dtype=np.float64)
    zero_points = np.asarray(zero_points)

    if q.size == 0:
        raise ValueError("Quantized tensor cannot be empty")

    if q.ndim == 0:
        raise ValueError("Quantized tensor must have at least one dimension")

    if not np.issubdtype(q.dtype, np.integer):
        raise TypeError("Quantized tensor must contain integers")

    if not isinstance(axis, (int, np.integer)):
        raise TypeError("Axis must be an integer")

    if not -q.ndim <= axis < q.ndim:
        raise ValueError("Invalid axis")

    axis = axis % q.ndim

    if scales.ndim != 1 or zero_points.ndim != 1:
        raise ValueError("Scales and zero points must be 1D arrays")

    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Scales must be positive and finite")

    if not np.issubdtype(zero_points.dtype, np.integer):
        raise TypeError("Zero points must be integers")

    if len(scales) != q.shape[axis] or len(zero_points) != q.shape[axis]:
        raise ValueError("Parameter count must match the selected axis")

    broadcast_shape = [1] * q.ndim
    broadcast_shape[axis] = len(scales)

    scales = scales.reshape(broadcast_shape)
    zero_points = zero_points.reshape(broadcast_shape)

    x_hat = scales * (q.astype(np.float64) - zero_points)

    return x_hat