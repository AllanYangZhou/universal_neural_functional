# Universal Neural Functionals

This is the code for constructing UNFs, from the paper *[Universal Neural Functionals](https://arxiv.org/abs/2402.05232).* UNFs are architectures that can process the weights of other neural networks, while maintaining equivariance or invariance to the weight space permutation symmetries.
In contrast to [NFNs](https://github.com/AllanYangZhou/nfn), UNFs can ingest weights from any architecture.

Equivalently, we can think of UNFs as equivariant architectures for processing any collection of tensors, where the action involves a shared set of permutations permuting the axes of the tensors in a given way.

The codebase requires JAX for core functionality and Flax for the example (though other Jax NN libraries are likely compatible as well). See usage in `example.py`.
