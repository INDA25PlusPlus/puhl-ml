# Simple Neural Network training on the MNIST dataset written in rust

## Iterations of network architecture
1. One hidden layer (128 wide) + ReLU + MSE + SGD: ~96% acc
2. One hidden layer (128 wide) + ReLU + **CrossEntropyLoss** + SGD: ~97.6% acc
3. **Two hidden layers (800, 300**) + ReLU + CrossEntropyLoss + SGD: ~98.24% acc
4. Two hidden layers (800, 300) + ReLU + CrossEntropyLoss + **SGD (with momentum)**: ~98.48% acc

## Notes
Every iteration of network architecture was trained on 50 epochs and with a 64 batch size.

## How to run
```console
$ cargo run --release
```

## Credit
Architectures based on them written at https://yann.lecun.org/exdb/mnist/
