from numpy.core.fromnumeric import var
from numpy.core.numeric import full
from z3 import *
from itertools import chain, product
import numpy as np


def createLayerVar(label: str, size: int):
    '''Return a layer of z3 Real number variables'''
    layer = [Real(f"{label}-{j}") for j in range(size)]
    return layer


def addLayerConstraints(solver: Solver, variables, nextLayerVariables, weight, bias, apply_relu=True):
    assert len(weight[0]) == len(variables), "Invalid variables!"
    assert len(weight) == len(nextLayerVariables), "Invalid next layer!"
    print(
        f"Adding constraints for layer of len {len(variables)} -> {len(nextLayerVariables)} with relu {apply_relu}")
    # calculate accumulated (i.e. pre-activation) values
    accumulated = [sum(v*m for v, m in zip(variables, mult)) +
                   additive for mult, additive in zip(weight, bias)]

    # add ReLU as a constraint
    if apply_relu:
        for acc, var in zip(accumulated, nextLayerVariables):
            solver.add(Or(And(acc > 0, var == acc), And(acc <= 0, var == 0)))
    else:
        for acc, var in zip(accumulated, nextLayerVariables):
            solver.add(var == acc)
    return


def createSolver(weights, biases):
    '''Creates z3 solver constraints
    Args:
        weights: A list of weights for each layer, corresponding to a feed-forward network with input weights. (Biases are not allowed!)
        Each weight in the list should be a 2-dimensional array of scalars.
    '''
    assert len(weights) == len(biases), "Invalid weights/biases!"
    s = Solver()
    inputSize = len(weights[0][0])
    inputVariables = createLayerVar("layer-0", inputSize)
    variables = inputVariables
    for layerIdx, (weight, bias) in enumerate(zip(weights, biases)):
        nextLayerVariables = createLayerVar(
            f"layer-{layerIdx+1}", len(weights[layerIdx]))
        addLayerConstraints(s, variables, nextLayerVariables,
                            weight, bias, layerIdx != len(weights) - 1)
        variables = nextLayerVariables

    # we return the solver, input layer vars, and output layer vars
    return s, inputVariables, variables


def testRobustness(solver: Solver, inputVariables, outputVariables, input, expected, eps):
    '''Tests for epsilon robustness of a single input/output pair.
    Note: This function is idempotent - you do not need to worry about cleaning up constraints
    Args:
        solver: The solver with created constraints
        inputVariables: The list of input variables returned by the solver creator
        outputVariables: The list of output variables returned by the solver creator
        input: The 784 len input image
        expected: The expected digit (i.e. pass in 7 if you expect it to be a 7)
        eps: The epsilon to test for
    '''
    # create new constraint frame
    solver.push()

    # add eps-closeness constraints to input vars (#TODO: make this frob. norm instead of elementwise?)
    for var, inp in zip(inputVariables, input):
        solver.add(And(var-inp >= 0, var-inp <= eps))

    # add max(outputs) != expected constraint
    solver.add(Or(*(outputVariables[expected] <= outVar for idx,
                    outVar in enumerate(outputVariables) if idx != expected)))
    sln = solver.check()
    model = None
    if sln == sat:
        model = solver.model()

    # revert added constraints
    solver.pop()

    return sln == sat, model
