# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List


def drop_duplicates(seq: List) -> List:
    """
    Returns a de-duplicates version of the input.
    Requires the input to be iterable.
    This method is necessary vs list(set(x)) as this doesn't keep the original order of the
    items.
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def cumsum(values: List[float], constant_delta: float = 0.0) -> List[float]:
    """
    Calculates the cumulative sum of the list/vector passed
    Doesn't change the values in place, instead return new instance
    """
    output = []
    last_value = 0
    for value in values:
        last_value = last_value + value
        output.append(last_value + constant_delta)
    return output


def lag(values: List[float], step: int, coalesce: float = None) -> List[float]:
    """
    Apply a lag to the values
    When there is no element before the step (e.g. i=0) then use coalesce value
    Doesn't change the values in place, instead return new instance
    """
    return [values[i - step] if i >=
            step else coalesce for i in range(len(values))]


def normalize(values: List[float]) -> List[float]:
    """
    Return a normalized version, where the highest value is 1
    """
    return [value / max(values) for value in values]
