from jutil import jtree, jmap
from jax import numpy as np
from jax import lax

@jtree
class Embedding:
    m: '(n, k)'
    _unq: '(n,)'

    def __call__(self, i: '(...)') -> '(..., k)':
        return self.m[i]

    @classmethod
    def from_unq(cls, unq: '(n,)', n_emb: int):
        """
        Initialize an embedding class capable of operating
        on the provided vector of unique symbols.
        """
        return cls(np.zeros((len(unq), n_emb)), unq)

@jtree
class Linear:
    m: '(n, k)'
    b: '(k,)'

    def __call__(self, v: '(..., n)') -> '(..., k)':
        return v @ self.m + self.b

    @classmethod
    def zeros(cls, n: int, k: int):
        """
        Initialize a linear class of a given size.
        """
        return cls(
            np.zeros((n, k)),
            np.zeros((k,)),
        )

    def balance(self):
        """
        Rebalance the weights so that a linear pass will preserve
        mean variance of a standard normal input.
        """
        n, _ = self.m.shape

        return type(self)(
            (self.m - np.mean(self.m, axis=0)) / np.maximum(np.sqrt(n *
                np.var(self.m, axis=0)), 1e-50),
            (self.b - np.mean(self.b)) / np.maximum(np.sqrt(np.var(self.b)),
                1e-50),
        )

@jtree
class Concat:
    _unused: None = None

    def __call__(self, *args):
        return np.concatenate([*args], axis=-1)

@jtree
class Chain:
    layers: list

    def __call__(self, *args):
        x = self.layers[0](*args)
        for f in self.layers[1:]:
            x = f(x)
        return x

def sigmoid(x):
    return (np.tanh(x)+1)/2

@jtree
class GRU:
    update: '(h,),(n,) -> (h,)'
    reset: '(h,),(n,) -> (h,)'
    activation: '(h,),(n,) -> (h,)'

    def __call__(self, h, x):
        """
        h: (k, h)
        x: (k, n)
        """
        z = jmap(sigmoid)(self.update(h, x))
        r = jmap(sigmoid)(self.reset(h, x))
        hhat = jmap(np.tanh)(self.activation(r*h, x))
        hnew = (1-z)*h+z*hhat
        return hnew

    def chain(self, h, x, n):
        def foo(z):
            return z, z

        return scan(
            lambda ht,_: foo(self(ht, x)),
            h,
            None,
            n
        )

    def scan(self, h, x):
        """
        x: (n_items, n_letters, n_dim)
        """
        def foo(z):
            return z, z

        return np.rollaxis(lax.scan(
            lambda carry, x: foo(self(carry, x)),
            h,
            np.rollaxis(x, 1),
        )[1], 1)
