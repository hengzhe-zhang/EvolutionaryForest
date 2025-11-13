"""Utility functions for compressing data to reduce multiprocessing transmission overhead."""

import gzip
import io
import pickle
import sys
from typing import Tuple, Any


def estimate_result_size(result: Tuple[Any, Any, Any]) -> float:
    """Estimate the total size in bytes of an evaluation result.

    Args:
        result: Tuple of (y_pred, estimators, info) from calculate_score

    Returns:
        Estimated size in bytes
    """
    total_size = 0
    y_pred, estimators, info = result

    # Estimate y_pred size
    if hasattr(y_pred, "nbytes"):
        total_size += y_pred.nbytes
    elif hasattr(y_pred, "__sizeof__"):
        total_size += sys.getsizeof(y_pred)

    # Estimate estimators size (can be large with CV)
    for est in estimators:
        if hasattr(est, "__sizeof__"):
            total_size += sys.getsizeof(est)

    # Estimate semantic_results size (often the largest)
    if info.semantic_results is not None:
        if hasattr(info.semantic_results, "nbytes"):
            total_size += info.semantic_results.nbytes
        elif hasattr(info.semantic_results, "__sizeof__"):
            total_size += sys.getsizeof(info.semantic_results)

    return total_size


def compress_result(
    result: Tuple[Any, Any, Any], threshold_mb: float = 1.0
) -> Tuple[Any, bool]:
    """Compress result to reduce multiprocessing transmission overhead.

    Compresses the whole result if total size > threshold, which is more efficient
    than selective compression when multiple components are large.

    Args:
        result: Tuple of (y_pred, estimators, info) from calculate_score
        threshold_mb: Size threshold in MB above which to compress (default: 1.0 MB)

    Returns:
        Tuple of (compressed_data, is_compressed) where:
        - compressed_data: Either compressed bytes or original result
        - is_compressed: Boolean flag indicating if data was compressed
    """
    # Estimate total size
    total_size = estimate_result_size(result)
    total_size_mb = total_size / (1024 * 1024)

    if total_size_mb > threshold_mb:
        # Compress the whole result
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=3) as f:
            pickle.dump(result, f, protocol=-1)
        return (buffer.getvalue(), True)  # Return compressed bytes + flag
    else:
        return (result, False)  # Return uncompressed + flag


def decompress_result(compressed_result: Tuple[Any, bool]) -> Tuple[Any, Any, Any]:
    """Decompress result after receiving from multiprocessing.

    Args:
        compressed_result: Tuple of (compressed_data, is_compressed) from compress_result

    Returns:
        Original result tuple (y_pred, estimators, info)
    """
    compressed_data, is_compressed = compressed_result

    if is_compressed:
        # Decompress the whole result
        buffer = io.BytesIO(compressed_data)
        with gzip.GzipFile(fileobj=buffer, mode="rb") as f:
            return pickle.load(f)
    else:
        # Already uncompressed
        return compressed_data
