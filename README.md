# Universal Neural Functionals

This is the code for constructing UNFs, from the paper *[Universal Neural Functionals](https://arxiv.org/abs/2402.05232).* UNFs are architectures that can process the weights of other neural networks, while maintaining equivariance or invariance to the weight space permutation symmetries.
In contrast to [NFNs](https://github.com/AllanYangZhou/nfn), UNFs can ingest weights from any architecture.

Equivalently, we can think of UNFs as equivariant architectures for processing any collection of tensors, where the action involves a shared set of permutations permuting the axes of the tensors in a given way.

The codebase requires JAX for core functionality and Flax for the example (though other Jax NN libraries are likely compatible as well). See usage in `example.py`.

## High level usage
The `perm_spec` is what tells our library the permutation symmetries it should be equivariant to. For example, suppose you have a collection of weight tensors corresponding to a simple MLP:
```python
params = {
    "params": {
        "Dense_0": {
            "kernel": Array[784, 512],
            "bias": Array[512]
        },
        "Dense_1": {
            "kernel": Array[512, 10],
            "bias": Array[10]
        }
    }
}
```
We can describe the permutation symmetry of this network as follows (assume the input and output neurons are also permutable).
* The weight tensors can be permuted by $\sigma=(\sigma_0, \sigma_1, \sigma_2) \in S_{784} \times S_{512} \times S_{10}$.
* $\sigma_0$ permutes the first dimension of `params["params"]["Dense_0"]["kernel"]`.
* $\sigma_1$ permutes the second dimension of `params["params"]["Dense_0"]["kernel"]`, the vector `params["params"]["Dense_0"]["bias"]`, and the first dimension of `params["params"]["Dense_1"]["kernel"]`.
* $\sigma_2$ permutes the second dimension of `params["params"]["Dense_1"]["kernel"]` and the vector `params["params"]["Dense_1"]["bias"]`.

Then we number each permutation by integers: $(\sigma_0, \sigma_1, \sigma_2) \mapsto (0, 1, 2)$ and define the permutation specification:
```python
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
```
Notice that nothing requires the input to be a collection of *weight* tensors. This library processes any collection of tensors if you give it a description of the permutation symmetries.
