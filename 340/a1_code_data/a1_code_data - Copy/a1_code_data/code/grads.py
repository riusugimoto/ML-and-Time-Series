import numpy as np


def example(x):
    return np.sum(x ** 2)


def example_grad(x):
    return 2 * x


def foo(x):
    result = 1
    λ = 4  # this is here to make sure you're using Python 3
    # ...but in general, it's probably better practice to stick to plaintext
    # names. (Can you distinguish each of λ𝛌𝜆𝝀𝝺𝞴 at a glance?)
    for x_i in x:
        result += x_i ** λ
    return result
 

def foo_grad(x):
     # Your implementation here...
    λ = 4
    return λ * x ** (λ - 1)


def bar(x):
    return np.prod(x)


def bar_grad(x):
    # For each i, the gradient element is the product of all x's except x_i.
    # This can be computed efficiently by computing the total product and then dividing by x[i].
    # However, if x[i] is 0, this approach won't work. So we need to handle that case separately.
    prod = np.prod(x)
    grad = []

    for x_i in x:
        if x_i == 0:
            # If x_i is zero, the derivative w.r.t. that element is the product of all other elements.
            grad.append(np.prod(x[x != 0]))
        else:
            grad.append(prod / x_i)

            
    return np.array(grad)