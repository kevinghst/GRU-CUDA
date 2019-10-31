# C++/CUDA Extensions in PyTorch

This repo contains CUDA implementations for various RNN architectures including:
- GRU
- GRU-tanh (all sigmoid replaced by tanh)

The CUDA implementations were guided by the following [tutorial](http://pytorch.org/tutorials/advanced/cpp_extension.html).

There are a few "sights" you can metaphorically visit in this repository:

- Navigate to the folder named after the model of interest
- Inspect the C++ and CUDA extensions in the `cpp/` and `cuda/` folders,
- Build C++ and/or CUDA extensions by going into the `cpp/` or `cuda/` folder and executing `python setup.py install`,
- Benchmark Python vs. C++ vs. CUDA by running `python benchmark.py {torch_lib, py, cpp, cuda} [--cuda]`,
- Run gradient checks on the code by running `python grad_check.py {py, cpp, cuda} [--cuda]`.
- Run output checks on the code by running `python check.py {forward, backward} [--cuda]`.
- Run sample training on MIMIC data and inspect training progression by running `python sample_train.py {torch_lib, py, cpp, cuda}`. 


## Compilation tips:
- At NYU's Prince HPC cluster, load the approriate gcc and cuda versions by executing `module load gcc/6.3.0`, and `module load cuda/10.1.105`
- Compile CUDA extension after sshing onto a GPU node. Re-compilation is neccessary whenever one ssh into a new GPU node.
- Remove **build**, **dist**, ***.egg-info** folders, and execute `python setup.py clean` between re-compilations

## Sample training:
The RNNs are tested on the MNIST dataset for classification.

The 28x28 MNIST images are treated as sequences of 28x1 vectors.

The RNN consist of

- A linear layer that maps 28-dimensional input to and 128-dimensional hidden layer
- One intermediate recurrent neural network
- A fully connected layer which maps the 128 dimensional input to 10-dimensional vector of class labels.
