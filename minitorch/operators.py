"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable
# Implementation of a prelude of elementary functions.


# Mathematical functions:
# - mul
def mul(x: float, y: float) -> float:
    return x * y


# - id  - Returns the input unchanged
def id(x: float):
    return x


# - add
def add(x: float, y: float) -> float:
    return x + y


# - neg
def neg(x: float) -> float:
    return -x


# - lt - Checks if one number is less than another
def lt(x: float, y: float) -> float:
    if x < y:
        return 1.0
    return 0.0


# - eq
def eq(x: float, y: float) -> float:
    if x == y:
        return 1.0
    return 0.0


# - max - Returns the larger of two numbers
def max(x: float, y: float) -> float:
    if x > y:
        return x
    return y


# - relu- Applies the ReLU activation function
def relu(x: float) -> float:
    if x > 0:
        return x
    return 0.0


EPS = 1e-6


# - log- Calculates the natural logarithm
def log(x: float) -> float:
    return math.log(x + EPS)


# - exp - Calculates the exponential function
def exp(x: float) -> float:
    return math.exp(x)


# - inv
def inv(x: float) -> float:
    return 1.0 / x


# - inv_back - Computes the derivative of reciprocal times a second arg
def inv_back(x: float, d: float) -> float:
    return -(1.0 / x**2) * d


# - log_back  - Computes the derivative of log times a second arg
def log_back(x: float, d: float) -> float:
    return d / (x + EPS)


# - relu_back  - Computes the derivative of ReLU times a second arg
def relu_back(x: float, d: float) -> float:
    if x > 0:
        return d
    return 0.0


# - sigmoid- Calculates the sigmoid function
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    return math.exp(x) / (1.0 + math.exp(x))


# For is_close: Checks if two numbers are close in value
# $f(x) = |x - y| < 1e-2$
def is_close(x: float, y: float) -> float:
    return (x - y < 1e-2) and (y - x < 1e-2)


# Implement for Task 0.1.


# ## Task 0.3

# Small practice library of elementary higher-order functions.
# Implement the following core functions


# - map
# Higher-order function that applies a given function to each element of an iterable
def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    def _map(ls: Iterable[float]) -> Iterable[float]:
        iter = []
        for x in ls:
            iter.append(fn(x))
        return iter

    return _map


# - zipWith
# Higher-order function that combines elements from two iterables using a given function
def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        iter = []
        for x, y in zip(ls1, ls2):
            iter.append(fn(x, y))
        return iter

    return _zipWith


# - reduce
# Higher-order function that reduces an iterable to a single value using a given function
def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    
    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for l in ls:
            val = fn(val, l)
        return val

    return _reduce


# Use these to implement
# - negList : negate a list
def negList(ls: Iterable[float]) -> Iterable[float]:
    return map(neg)(ls)


# - addLists : add two lists together
# Add corresponding elements from two lists, ls1 and ls2, by using zipWith
def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    return zipWith(add)(ls1, ls2)


# - sum: sum lists
def sum(ls: Iterable[float]) -> float:
    return reduce(add, 0.0)(ls)


# - prod: take the product of lists
def prod(ls: Iterable[float]) -> float:
    return reduce(mul, 1.0)(ls)


# TODO: Implement for Task 0.3.
