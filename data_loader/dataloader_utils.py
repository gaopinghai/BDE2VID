import numpy as np


def first_element_greater_than(values, req_value):
    """Returns the pair (i, values[i]) such that i is the minimum value that satisfies values[i] >= req_value.
    Returns (-1, None) if there is no such i.
    Note: this function assumes that values is a sorted array!"""
    i = np.searchsorted(values, req_value)
    val = values[i] if i < len(values) else None
    return (i, val)


def last_element_less_than(values, req_value):
    """Returns the pair (i, values[i]) such that i is the maximum value that satisfies values[i] <= req_value.
    Returns (-1, None) if there is no such i.
    Note: this function assumes that values is a sorted array!"""
    i = np.searchsorted(values, req_value, side='right') - 1
    val = values[i] if i >= 0 else None
    return (i, val)