# jutil

A small set of utilities for jax

## Purpose

The main contribution is `jtree` which has some boilerplate for making
dataclasses differentiable. I've written something like that dozens of times
now and wanted to put some version of it somewhere concrete.

## Status

Early alpha. `Linear.balance` is generating nans occasionally and might be
broken. Everything else seems to work fine. This isn't a high priority for me,
but I'll probably fix it up and improve the docs at some point.

## Installation

```bash
python -m pip install -e git+https://github.com/hmusgrave/jutil.git#egg=jutil
```

## Examples

```
from jutil import jtree, keygen, randomize
from jax import random

"""
Seed a stateful PRNG with 42 and use it to
draw 60 (3 * 4 * 5) standard normal values.
"""
new_key = keygen(42)
experiment = random.normal(new_key(), (3, 4, 5))

@jtree
class Linear:
    """
    Initialized much like a dataclass. Jax propagates
    gradient information to most fields.
    """
    m: '(n, k)'
    b: '(k,)'

    """
    Prefix fields with an underscore to indicate that
    they shouldn't be used in gradient computations
    and other arithmetic.
    """
    _name: str

    def __call__(self, v: '(..., n)') -> '(..., k)':
        return v @ self.m + self.b

"""
A few simple types of operations like that are already
included in jutil.
"""
from jutil.model import Linear

clf = Linear(np.ones((3, 7)), np.ones((7,)))
clf = randomize(new_key(), clf, random.normal)

"""
All basic arithmetic is supported. At this point, clf
has each element initialized to a gaussian with mean
12 and variance 1.
"""
clf += 12.
```
