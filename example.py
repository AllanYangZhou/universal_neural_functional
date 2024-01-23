import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import flax.linen as nn

from layers import UNFLayer, UNFSequential, nf_relu


class MLP(nn.Module):
    """Two layer MLP"""
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(128)(x))
        return nn.Dense(10)(x)


# Label the dims of each parameter with a permutation index. We count the permutations
# 0, 1, 2, ... but any distinct integers work.
perm_spec = {
    "params": {
        "Dense_0": {
            "kernel": (0, 1),
            "bias": (1,)
        },
        "Dense_1": {
            "kernel": (1, 2),
            "bias": (2,)
        }
    }
}

key = jax.random.PRNGKey(0)
mlp_params = MLP().init(key, jnp.ones((64)))

unf = UNFSequential([
    UNFLayer(16, 1),
    nf_relu,
    UNFLayer(16, 16),
])

# add channel dim to input
mlp_params = jtu.tree_map(lambda x: jnp.expand_dims(x, -1), mlp_params)
print(jtu.tree_map(jnp.shape, mlp_params))

unf_params = unf.init(key, mlp_params, perm_spec)
# apply unf to input
unf_out = unf.apply(unf_params, mlp_params, perm_spec)
print(jtu.tree_map(jnp.shape, unf_out))
