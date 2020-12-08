# NN-SAT-Verifier
Verifying epsilon robustness of neural networks using SAT solvers

This is a project that aims to demonstrate the generation of adversarial examples for a neural network based classifier.
The file structure is as follows:
analysis/verification contains the z3 solver code. To use the verifier directly, see the `RobustnessChecker` class (documentation should be sufficient).

If you wish to simply run experiments, then run the scripts in the following order:

1) Run `train.ipynb` to train a classifier in PyTorch. The weights will be saved in the weights folder. One can change which dataset to use by editing the 'dataset' variable in the notebook.

2) Next, run `generate_adv_examples.py` to run the solver and to generate adversarial examples. The adversarial examples will be saved  in the adv_examples folder.
By default, the script looks for weights corresponding to the SVHN dataset, with hidden layer size 15, and 40 epochs of training. This can be changed by updating the `weight_path` argument of the function call in `generate_adv_examples.py`, at line 109.