import typing as tp
import collections
import itertools
import jax
import jax.numpy as jnp


class LeafTuple(tuple):
    """Custom tuple class treated as a leaf by Jax pytree utils."""

    def __new__(cls, *args):
        return tuple.__new__(cls, args)


def group_indices_by_perms(indices, perms):
    """Group indices by corresponding values in perms."""
    groups = collections.defaultdict(list)
    for index, perm in zip(indices, perms):
        groups[perm].append(index)
    return groups


def generate_partitions(items):
    """Generate all non-empty partitions of items."""
    if len(items) == 1:
        yield [items]
        return
    first = items[0]
    for subpart in generate_partitions(items[1:]):
        for n, subset in enumerate(subpart):
            yield subpart[:n] + [[first] + subset] + subpart[n + 1 :]
        yield [[first]] + subpart


def valid_partitions(indices, perms):
    """Only partitions of indices such that subsets share value in perms."""
    # Create a mapping from perm to indices
    groups_map = group_indices_by_perms(indices, perms)
    groups = list(groups_map.values())

    # Generate all possible products of non-empty subsets of the groups
    partitions = [
        itertools.chain(*p)
        for p in itertools.product(*[generate_partitions(g) for g in groups])
    ]

    for partition in partitions:
        yield [p for p in partition]


def get_repeated_idcs(lst):
    """Get indices of list for values that repeat."""
    index_map = {}
    for index, item in enumerate(lst):
        if item in index_map:
            index_map[item].append(index)
        else:
            index_map[item] = [index]
    rep_idcs = []
    for idcs in index_map.values():
        if len(idcs) > 1:
            rep_idcs.append(idcs)
    return rep_idcs


def diagonal_slice(X, matching_dims):
    """Take X's diagonal along some subset of dimensions."""
    # len(matching_dims) must be at least 2
    out = jnp.diagonal(X, axis1=matching_dims[0], axis2=matching_dims[1])
    for dim in matching_dims[2:]:
        # The diagonal dim is always kept at the end.
        out = jnp.diagonal(out, axis1=dim, axis2=-1)
    return out


def move_dims_to_end(X, dims):
    """Move the specified dims of X to the end."""
    new_in_order = remove_idcs(list(range(X.ndim)), dims) + dims
    return jnp.transpose(X, new_in_order)


def remove_idcs(lst, idcs):
    return [l for idx, l in enumerate(lst) if idx not in idcs]


def find_missing_indices(lstA, lstB):
    """Find the indices for elements of A not in B."""
    setB = set(lstB)
    missing = []
    for i, el in enumerate(lstA):
        if el not in setB:
            missing.append(i)
    return missing


def is_broadcastable(shp1, shp2):
    """Check two array shapes are broadcastable."""
    for a, b in zip(shp1[::-1], shp2[::-1]):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            return False
    return True


def match_dim_order(X, X_idcs, out_idcs):
    """Given X and idcs naming each dimension, reorder them to match out_idcs."""
    X_idcs_copy = X_idcs[:]
    perm = []
    for idx in out_idcs:
        if idx in X_idcs_copy:
            i = X_idcs_copy.index(idx)
            perm.append(i)
            X_idcs_copy[i] = None  # "use up" this index.
    X = jnp.transpose(X, perm)
    X_idcs = [X_idcs[i] for i in perm]
    return X, X_idcs


def gen_basis(
        out_spec: tp.Tuple[int], in_spec: tp.Tuple[int], out_shape: tp.Tuple[int], inp: jax.Array
    ) -> tp.List[jax.Array]:
    """Generate the basis terms of an equivariant map from in_spec to out_spec.

    Args:
        out_spec: specifies which permutation affects each axis of the output (except final axis, which is channels)
        in_spec: specifies which permutation affects each axis of the input (except final axis, which is channels)
        out_shape: shape of output tensor
        inp: input tensor
    """
    terms = []
    k1, k2 = len(out_spec), len(in_spec)  # number of output/input indices
    for partition in valid_partitions(list(range(k1 + k2)), out_spec + in_spec):
        indices = [-1] * (k1 + k2)
        for subset_idx, subset in enumerate(partition):
            for dim in subset:
                indices[dim] = subset_idx
        out_indices, in_indices = indices[:k1], indices[k1:]
        term = inp
        # For repeated input idcs, take the corresponding diagonals
        while rep_idcs := get_repeated_idcs(in_indices):
            term = diagonal_slice(term, rep_idcs[0])
            # Diagonal is a new dim at the end
            idx = in_indices[rep_idcs[0][0]]
            in_indices = remove_idcs(in_indices, rep_idcs[0])
            in_indices.append(idx)
        # Aggregate over in_indices that aren't in out_indices
        agg_idcs = find_missing_indices(in_indices, out_indices)
        term = jnp.mean(term, axis=agg_idcs)
        in_indices = remove_idcs(in_indices, agg_idcs)
        # If we are outputting onto a diagonal, construct a larger tensor and then
        # embed term into the appropriate diagonal.
        for diag_set in get_repeated_idcs(out_indices):
            dim_size, out_idx = out_shape[diag_set[0]], out_indices[diag_set[0]]
            matching_in_dims = [
                i for i, idx in enumerate(in_indices) if idx == out_idx
            ]
            num_new_dims = len(diag_set) - len(matching_in_dims)
            # move matching in dimensions to end
            term = move_dims_to_end(term, matching_in_dims)
            new_term = jnp.zeros(term.shape + (dim_size,) * num_new_dims)
            diag_slice = (jnp.s_[:],) * (new_term.ndim - len(diag_set)) + (
                jnp.arange(dim_size),
            ) * len(diag_set)
            if not matching_in_dims:
                term = jnp.expand_dims(term, axis=-1)
            term = new_term.at[diag_slice].add(term)
            in_indices = remove_idcs(in_indices, matching_in_dims)
            in_indices.extend([out_idx] * len(diag_set))
        # Permute term dimensions to match out_indices.
        term, in_indices = match_dim_order(term, in_indices, out_indices)
        # Broadcast
        term = jnp.expand_dims(
            term, axis=find_missing_indices(out_indices, in_indices)
        )
        terms.append(term)
    return terms
