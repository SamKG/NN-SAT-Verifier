import re

from z3 import *


def multiplyLayer(solver: Solver, variables, weight, bias):
    assert len(weight[0]) == len(variables), "Invalid variables!"
    assert len(weight) == len(bias), "Invalid bias!"
    print(
        f"Adding constraints for layer of len {len(variables)} -> {len(weight)}")
    # calculate accumulated (i.e. pre-activation) values
    accumulated = [sum(v*m for v, m in zip(variables, mult)) +
                   additive for mult, additive in zip(weight, bias)]

    return accumulated


def abs(var):
    '''Returns absolute value of var'''
    return If(var >= 0, var, -var)


def relu(var):
    '''Returns relu of var'''
    return If(var >= 0, var, 0)


def applyRelu(solver: Solver, variables):
    activated = [relu(v) for v in variables]

    return activated


class RobustnessChecker:
    def __init__(self, weights, biases):
        '''Creates a robustness checker
        Args:
            weights: A list of weights for each layer, corresponding to a feed-forward network with input weights. (Biases are not allowed!)
            Each weight in the list should be a 2-dimensional array of scalars.
        '''
        assert len(weights) == len(biases), "Invalid weights/biases!"

        s = Solver()
        inputSize = len(weights[0][0])
        inputVariables = [Real(f"input-{j}") for j in range(inputSize)]

        variables = inputVariables
        for layerIdx, (weight, bias) in enumerate(zip(weights, biases)):
            variables = multiplyLayer(s, variables,
                                      weight, bias)
            if layerIdx != len(weights) - 1:
                variables = applyRelu(s, variables)

        self._inpVar = inputVariables
        self._outVar = variables
        self._solver = s

    def testInputRobustness(self, input, expected, delta=1):
        '''Tests for epsilon robustness of a single input/output pair.
        Note: This function is idempotent - you do not need to worry about cleaning up constraints
        Args:
            input: The 784 len input image
            expected: The expected digit (i.e. pass in 7 if you expect it to be a 7)
            delta: The max change in a single pixel of input
        '''
        # create new constraint frame
        self._solver.push()

        # add eps-closeness constraints to input vars (#TODO: make this frob. norm instead of elementwise?)
        for var, inp in zip(self._inpVar, input):
            self._solver.add(abs(var-inp) <= delta)

        # add max(outputs) != expected constraint
        self._solver.add(Or(*(self._outVar[expected] <= outVar for idx,
                              outVar in enumerate(self._outVar) if idx != expected)))

        sln = self._solver.check()
        model = None
        if sln == sat:
            def getInt(a):
                return int(re.search(r"\d+", a.name())[0])
            model = [float(model[v].as_decimal(10))
                     for v in sorted(self._solver.model(), key=getInt)]

        # revert added constraints
        self._solver.pop()

        return sln == sat, model

    def testCorrectness(self, input, output, tol=1E-5):
        '''Tests for correctness of the implementation by comparing inputs and outputs
        of a passed value
        Args:
            input: The 784 len input image
            output: The expected output vector
            tol: The max tolerance on outputs (needed to fix numerical precision issues)
        '''
        assert len(self._inpVar) == len(input), "Invalid input!"
        assert len(self._outVar) == len(output), "Invalid output!"

        self._solver.push()

        # add passed inputs
        for ivar, inp in zip(self._inpVar, input):
            self._solver.add(ivar == inp)

        for ovar, out in zip(self._outVar, output):
            self._solver.add(abs(ovar-out) <= tol)

        sln = self._solver.check()
        assert sln == sat, "The inputs and outputs did not match! Please check implementation for correctness!"

        self._solver.pop()
