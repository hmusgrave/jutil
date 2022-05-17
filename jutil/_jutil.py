from dataclasses import dataclass
from jax import random
from jax.tree_util import register_pytree_node, tree_map

def jtree(cls, *dargs, **dkwargs):
    cls = dataclass(cls, *dargs, **dkwargs)

    fields = list(cls.__dataclass_fields__)
    gradient_fields = sorted(x for x in fields if not x.startswith('_'))
    other_fields = sorted(set(fields) - set(gradient_fields))

    def flatten(self):
        children = tuple(getattr(self, s) for s in gradient_fields)
        aux = tuple(getattr(self, s) for s in other_fields)
        return (children, aux)

    def unflatten(aux, children):
        return cls(
            **dict(zip(other_fields, aux)),
            **dict(zip(gradient_fields, children))
        )

    register_pytree_node(cls, flatten, unflatten)

    def op(f):
        def _op(self, *others):
            # Admits a limited form of broadcasting
            if others and isinstance(others[0], type(self)):
                return tree_map(f, self, *others)
            return tree_map(lambda a: f(a, *others), self)
        return _op

    for s,b in [
        ('add', '+'), ('sub', '-'), ('mul', '*'), ('truediv', '/'),
        ('floordiv', '//'), ('mod', '%'), ('pow', '**'), ('lshift', '<<'),
        ('rshift', '>>'), ('and', '&'), ('xor', '^'), ('or', '|')
    ]:
        f = op(eval(f'lambda x,y: x{b}y'))
        setattr(cls, f'__{s}__', f)
        setattr(cls, f'__r{s}__', f)

    for s,b in [('neg', '-'), ('pos', '+'), ('invert', '~')]:
        f = op(eval(f'lambda x: {b}x'))
        setattr(cls, f'__{s}__', f)

    return cls

def randomize(key, tree, distribution):
    def split():
        nonlocal key
        key, rtn = random.split(key)
        return rtn
    return tree_map(lambda a: distribution(split(), a.shape), tree)

def jmap(tree, f, *a, **k):
    return tree_map(f, tree, *a, **k)

def keygen(seed):
    key = random.PRNGKey(seed)
    def gen():
        nonlocal key
        key, rtn = random.split(key)
        return rtn
    return gen
