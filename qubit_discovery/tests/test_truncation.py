"""Test Module for truncation module functionalities."""
import numpy as np

from qubit_discovery.optimization.truncation import (
    get_indices,
    filter_vec,
    check_convergence,
)


def test_indices() -> None:
    """Test get_sub_indices."""

    m_shrank = [1, 1, 1]
    m = [2, 2, 1]

    indices = get_indices(m_shrank, m)

    assert np.array_equal(np.array(indices), np.array([1, 2, 3]))


def test_filter_vec() -> None:
    """Test filter_vec."""

    vec = np.array([[1], [2], [3]])
    expected_filtered_vec = np.array([[0], [0], [3]])

    filtered_vec = filter_vec(vec, indices=[0, 1])

    assert np.array_equal(filtered_vec, expected_filtered_vec)


def test_check_convergence() -> None:
    """Test check_convergence."""

    assert True



