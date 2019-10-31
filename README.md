# C++/CUDA Extensions in PyTorch

An example of writing a C++ extension for PyTorch. See
[here](http://pytorch.org/tutorials/advanced/cpp_extension.html) for the accompanying tutorial.

This repo contains CUDA implementations for various RNN architectures including:
- GRU
- GRU-tanh (all sigmoid replaced by tanh)

The implementations were guided by the [following tutorial](http://pytorch.org/tutorials/advanced/cpp_extension.html).

There are a few "sights" you can metaphorically visit in this repository:

- Navigate to the folder named after the model of interest
- Inspect the C++ and CUDA extensions in the `cpp/` and `cuda/` folders,
- Build C++ and/or CUDA extensions by going into the `cpp/` or `cuda/` folder and executing `python setup.py install`,
- Benchmark Python vs. C++ vs. CUDA by running `python benchmark.py {torch_lib, py, cpp, cuda} [--cuda]`,
- Run gradient checks on the code by running `python grad_check.py {py, cpp, cuda} [--cuda]`.
- Run output checks on the code by running `python check.py {forward, backward} [--cuda]`.
- Run sample training on MIMIC data and inspect training progression by running `python sample_train.py {torch_lib, py, cpp, cuda}`


## Compilation tips:
- At NYU's Prince HPC cluster, load the approriate gcc and cuda versions by executing `module load gcc/6.3.0`, and `module load cuda/10.1.105`
- Compile CUDA extension after sshing onto a GPU node. Re-compilation is neccessary whenever one ssh into a new GPU node.
- Remove **build**, **dist**, ***.egg-info**, and execute `python setup.py clean` between re-compilations
