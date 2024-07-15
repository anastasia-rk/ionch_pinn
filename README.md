Attept at solving ion channel models with Physics-Informed Neural Networks.

Current issues: 
1. Speed of convergence - a simple 2-state model takes 400 epochs to train.
2. Initialisation - standard initialisation of weights and biases is generated from uniform distribution in the interval [-1, 1]. It seems that extending the range for biases will help cover the entire state space on the domain.
3. If we have multiple inputs, gradient computation becomes tricky. Need to explore altrernative implementations of derivative computation, maybe using torch func
4. What about other activators? Need to see if Leaky ReLU would be better/more efficient.

