# borrowed from http://proximity-operator.net/

from typing import Union, Optional
import jax.numpy as np


class L2Norm:
    r"""Compute the proximity operator and the evaluation of gamma*f.

    Where f is the euclidian norm:

                        f(x) = ||x||_2

    'gamma' is the scale factor

     INPUTS
    ========
     x         - ND array
     gamma     - positive, scalar or ND array compatible with the blocks of 'x'
                 [default: gamma=1]
     axis    - None or int, axis of block-wise processing [default: axis=None]
                  axis = None --> 'x' is processed as a single vector [DEFAULT] In this
                  case 'gamma' must be a scalar.
                  axis >=0   --> 'x' is processed block-wise along the specified axis
                  (0 -> rows, 1-> columns ect. In this case, 'gamma' must be singleton
                  along 'axis'.
    """

    def __init__(
            self,
            axis: Optional[int] = None
    ):
        self.axis = axis

    def prox(self, x: np.ndarray, gamma: Union[float, np.ndarray] = 1.0) -> np.ndarray:
        l2_x2 = np.sqrt(np.sum(x ** 2, axis=self.axis, keepdims=True))
        eps = 1e-16  # to avoid dividing by zeros
        l2_x2 = np.maximum(0, 1 - gamma / (eps + l2_x2))
        prox_x = x * l2_x2
        return prox_x

    def __call__(self, x: np.ndarray) -> float:
        l2_x = np.sqrt(np.sum(x ** 2, axis=self.axis))
        return np.sum(l2_x)



class L21columns:
    r"""Compute the proximity operator and the evaluation of gamma*f.

     Where f is the sum of euclidian norms of the columns of a matrix:

             f(x) = \sum_{j= 1}^M |\sum_{i=1}^N |X(i,j)|^2|^{\frac{1}{2}}

            where X = U*diag(s)*V.T \in R^{M*N}  (Singular Value decomposition)

     INPUTS
    ========
     x         -  (M,N) -array_like ( representing an M*N matrix )
     gamma     - positive, scalar or ND array compatible with the size of 'x'
                 [default: gamma=1]
    """

    def __init__(self):
        pass

    def prox(self, x: np.ndarray, gamma: Union[float, np.ndarray] = 1.0) -> np.ndarray:
        return L2Norm(axis=0).prox(x, gamma)

    def __call__(self, x: np.ndarray) -> float:
        return L2Norm(axis=0)(x)


class Thresholder:
    r"""Computes the proximity operator and the evaluation of gamma*f.

    Where f is the Thresholder (or Support) function defined as:

                    /  a * x   if x < 0
            f(x) = |   0                        if x = 0         with a <= b
                   \  b * x    otherwise

     'gamma' is the scale factor

    When the input 'x' is an array, the output is computed element-wise :

    -When calling the function, the output is a scalar (sum of the
    element-wise results ) .

    - But for the proximity operator (method 'prox'), the output has the same
    shape as the input 'x'.

     INPUTS
    ========
     x     - scalar or ND array
     a     - scalar or ND array with the same size as 'x' [default: a=-1]
     b     - scalar or ND array with the same size as 'x' [default: b=1]
     gamma - positive, scalar or ND array with the same size as 'x' [default: gamma=1.0]

     ========
     Examples
     ========

     Evaluate the 'direct' function f (i.e compute f(x)  ):

     >>> Thresholder(-2, 2)(3)
     6
     >>> Thresholder(-2, 2, gamma=2)([3, 4, -2])
     36

     Compute the proximity operator at a given point :

     >>> Thresholder(-2, 2).prox( 3)
     1
     >>> Thresholder(-1, 2).prox([ -3., 1., 6.])
     array([-2.,  0.,  4.])

     Use a scale factor 'gamma'>0 to commute the proximity operator of gamma*f

     >>> Thresholder(-1, 2).prox([ -3., 1., 6.], gamma=2)
      array([-1.,  0.,  2.])
    """

    def __init__(
        self,
        a: Union[float, np.ndarray] = -1,
        b: Union[float, np.ndarray] = 1
    ):
        self.a = a
        self.b = b

    def prox(self, x: np.ndarray, gamma: Union[float, np.ndarray] = 1.0) -> np.ndarray:
        return np.minimum(0, x - self.a * gamma) + np.maximum(0, x - self.b * gamma)

    def __call__(self, x: np.ndarray) -> float:
        return np.sum((self.a * np.minimum(0, x) + self.b * np.maximum(0, x)))


class AbsValue(Thresholder):
    def __init__(self):
        super().__init__(-1, 1)

def prox_svd(x, gamma, prox_phi, hermitian=False):
    r"""Compute the proximity operator of a matrix function.

                        f(X) = gamma * prox_phi(s)

    Where X = U*diag(s)*V.T \in R^{M*N}  is the Singular Value decomposition of X

      INPUTS
     ========
    x       - ND array
    gamma   - positive, scalar or ND array compatible with the size of 'x'
    prox_phi - function handle with two arguments at least
    """
    # spectral decomposition
    u, s, vh = np.linalg.svd(x, full_matrices=False, hermitian=hermitian)
    # prox computation
    g = np.reshape(prox_phi(s, gamma), np.shape(s))
    return np.matmul(u, g[..., None] * vh)


class NuclearNorm:
    r"""Compute the proximity operator and the evaluation of gamma*f.

    Where f is the function defined as:


                        f(x) = ||X||_N = ||s||_1

            where X = U*diag(s)*V.T \in R^{M*N}

     INPUTS
    ========
     x         - (M,N) -array_like ( representing an M*N matrix )
     gamma     - positive, scalar or ND array compatible with the size of 'x'
    """

    def __init__(self):
        pass

    def prox(self, x: np.ndarray, gamma: Union[float, np.ndarray] = 1) -> np.ndarray:
        return prox_svd(x, gamma, AbsValue().prox)

    def __call__(self, x: np.ndarray) -> float:
        return np.linalg.norm(x, ord='nuc')