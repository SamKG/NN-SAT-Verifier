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


def z3abs(var):
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
        self.weights = weights
        self.biases = biases

    def _createSolver(self, inputs=[None]*28*28):
        weights, biases = self.weights, self.biases
        inputSize = len(weights[0][0])
        assert len(inputs) == inputSize, "Invalid input length!"

        s = Solver()
        inputVariables = [Real(f"input-{j}") if inp is None else inp.item() for j, inp in enumerate(inputs)]
        variables = inputVariables
        for layerIdx, (weight, bias) in enumerate(zip(weights, biases)):
            variables = multiplyLayer(s, variables,
                                      weight, bias)
            if layerIdx != len(weights) - 1:
                variables = applyRelu(s, variables)
        return s, inputVariables, variables

    def testInputRobustness(self, inputs, expected, delta=1):
        '''Tests for epsilon robustness of a single input/output pair.
        Note: This function is idempotent - you do not need to worry about cleaning up constraints
        Args:
            input: The 784 len input image
            expected: The expected digit (i.e. pass in 7 if you expect it to be a 7)
            delta: The max change in a single pixel of input
        '''
        solver, inputVariables, outputVariables = self._createSolver(inputs=[inp if abs(inp - -0.42421296) < 1E-5 else None for inp in inputs])

        for idx, (var, inp) in enumerate(zip(inputVariables, inputs)):
            if abs(inp - -0.42421296) >= 1E-5:
                solver.add(var <= inp+delta, var >= inp-delta)

        # add max(outputs) != expected constraint
        solver.add(Or(*(simplify(outputVariables[expected] < outVar) for idx,
                              outVar in enumerate(outputVariables) if idx != expected)))

        sln = solver.check()
        model = None
        if sln == sat:
            model = solver.model()
            def getInt(a):
                return int(re.search(r"\d+", a.name())[0])
            def toFloat(a):
                if a[-1] == '?':
                    return float(a[:-1])
                return float(a)

            model = [toFloat(model[v].as_decimal(10))
                     for v in sorted(model, key=getInt)]

        return sln == sat, model

    def testOnePixelInputRobustness(self, inputs, expected, pixel_idx=0, delta=1):
        '''Tests for epsilon robustness of a single input/output pair.
        Note: This function is idempotent - you do not need to worry about cleaning up constraints
        Args:
            input: The 784 len input image
            expected: The expected digit (i.e. pass in 7 if you expect it to be a 7)
            delta: The max change in a single pixel of input
        '''
        solver, inputVariables, outputVariables = self._createSolver(inputs=[inp if idx != pixel_idx else None for idx, inp in enumerate(inputs)])
        
        # add eps-closeness constraints to input vars (#TODO: make this frob. norm instead of elementwise?)
        for idx, (var, inp) in enumerate(zip(inputVariables, inputs)):
            if idx == pixel_idx:
                solver.add(var <= inp+delta, var >= inp-delta)

        # add max(outputs) != expected constraint
        solver.add(Or(*(simplify(outputVariables[expected] < outVar) for idx,
                              outVar in enumerate(outputVariables) if idx != expected)))

        sln = solver.check()
        model = None
        if sln == sat:
            model = solver.model()
            def getInt(a):
                return int(re.search(r"\d+", a.name())[0])
            def toFloat(a):
                if a[-1] == '?':
                    return float(a[:-1])
                return float(a)

            model = [toFloat(model[v].as_decimal(10))
                     for v in sorted(model, key=getInt)]

        return sln == sat, model
    
    def testAllOnePixelInputRobustness(self, input, expected, delta):
        for i in range(len(input)):
            isSat, model = self.testOnePixelInputRobustness(input, expected, i, delta)
            if isSat:
                break
            
        return isSat, model

    def testCorrectness(self, input, output, tol=1E-5):
        '''Tests for correctness of the implementation by comparing inputs and outputs
        of a passed value
        Args:
            input: The 784 len input image
            output: The expected output vector
            tol: The max tolerance on outputs (needed to fix numerical precision issues)
        '''
        solver, inputVariables, outputVariables = self._createSolver(inputs=input)
        assert len(inputVariables) == len(input), "Invalid input!"
        assert len(outputVariables) == len(output), "Invalid output!"

        for ovar, out in zip(outputVariables, output):
            solver.add(z3abs(ovar-out) <= tol)

        sln = solver.check()
        assert sln == sat, "The inputs and outputs did not match! Please check implementation for correctness!"
