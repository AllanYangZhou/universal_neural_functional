import typing as tp
import math
import functools
import flax.linen as nn
import jax
import jax.random as jrandom
import jax.tree_util as jtu

from algorithm import gen_basis, LeafTuple


def build_init_fn(scale, shape):
    return lambda rng, _shape: scale * jrandom.normal(rng, shape)


class UNFLayer(nn.Module):
    """Linear equivariant layer for an arbitrary weight space."""

    c_out: int
    c_in: int

    @nn.compact
    def __call__(self, params, perm_spec):
        """perm_spec matches the tree structure of param, and associates each parameter with a tuple of
            integers specifying how the dimensions of that parameter are be permuted. For example, if we
            have `params={"W": jnp.ones((5, 5))}` and both rows and columns are permuted simultaneously,
            we have `perm_spec={"W": (0, 0)}`. If the rows and columns can permute simultaneously, we
            instead have `perm_spec={"W": (0, 1)}`.
        """
        params_and_spec, tree_def = jtu.tree_flatten(
            jtu.tree_map(LeafTuple, params, perm_spec)
        )
        flat_params = [x[0] for x in params_and_spec]
        flat_spec = [x[1] for x in params_and_spec]
        L = len(flat_params)

        outs = []
        for i in range(L):  # output
            terms_i = []
            for j in range(L):  # input
                in_param, out_param = flat_params[j], flat_params[i]
                out_spec, in_spec = flat_spec[i], flat_spec[j]
                term_gen_fn = jax.vmap(
                    functools.partial(gen_basis, out_spec, in_spec, out_param.shape),
                    in_axes=-1,
                    out_axes=-1,
                )
                terms_i.extend(term_gen_fn(in_param))
            fan_in = self.c_in * len(terms_i)
            scale = math.sqrt(1 / fan_in)
            shape = (len(terms_i), self.c_in, self.c_out)
            theta_i = self.param(f"theta_{i}", build_init_fn(scale, shape), shape)
            out = 0
            for j, term in enumerate(terms_i):
                out += term @ theta_i[j]
            bias_i = self.param(
                f"bias_{i}", nn.initializers.zeros_init(), (self.c_out,)
            )
            out += bias_i
            outs.append(out)

        return jtu.tree_unflatten(tree_def, outs)


def nf_relu(params):
    """Apply relu to each weight-space feature."""
    return jtu.tree_map(nn.relu, params)


# Define batched layers
BatchUNFLayer = nn.vmap(
    UNFLayer,
    in_axes=(0, None),
    out_axes=0,
    variable_axes={"params": None},
    split_rngs={"params": False},
)


class UNFSequential(nn.Module):
    layers: tp.List  # pylint: disable=g-bare-generic

    @nn.compact
    def __call__(self, params, spec):
        out = params
        for layer in self.layers:
            if isinstance(layer, UNFLayer):
                out = layer(out, spec)
            else:
                out = layer(out)
        return out
