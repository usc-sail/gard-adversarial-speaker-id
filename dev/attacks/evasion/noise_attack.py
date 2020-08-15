"""
This module implements a "attack" that injects noise. 
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from art.config import ART_NUMPY_DTYPE
from art.classifiers.classifier import ClassifierGradients
from art.attacks.attack import EvasionAttack
from art.utils import compute_success, get_labels_np_array, random_sphere, projection, check_and_transform_label_format
from art.exceptions import ClassifierError

logger = logging.getLogger(__name__)


class NoiseAttack(EvasionAttack):
    """ Inject noise to the input signal (as a baseline) """
    attack_params = EvasionAttack.attack_params + [
        "norm",
        "eps",
        "targeted",
        "batch_size",
    ]

    def __init__(
        self,
        classifier,
        norm=np.inf,
        eps=1e-1,
        targeted=False,
        batch_size=1,
    ):
        super().__init__(classifier)
        if not isinstance(classifier, ClassifierGradients):
            raise ClassifierError(self.__class__, [ClassifierGradients], classifier)
        
        kwargs = {
            "norm": norm,
            "eps": eps,
            "targeted": targeted,
            "batch_size": batch_size,
        }

        NoiseAttack.set_params(self, **kwargs)

        self._project = True

    
    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param norm: Order of the norm. Possible values: np.inf, 1 or 2.
        :type norm: `int` or `float`
        :param eps: Attack step size (input variation)
        :type eps: `float`
        :param targeted: Should the attack target one specific class
        :type targeted: `bool`
        :param batch_size: Batch size
        :type batch_size: `int`
        """
        super().set_params(**kwargs)

        if self.norm not in [np.inf]:
            raise ValueError("Norm order must be either `np.inf`, 1, or 2.")

        if self.eps <= 0:
            raise ValueError("The perturbation size `eps` has to be positive.")

        if self.batch_size <= 0:
            raise ValueError("The batch size `batch_size` has to be positive.")

        if not isinstance(self.targeted, bool):
            raise ValueError("The flag `targeted` has to be of type bool.")

        if self.targeted is True:
            logger.warn("`targeted` in NoiseAttack has no effects.")


    def generate(self, x, y=None, **kwargs):
        """Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :type x: `np.ndarray`
        :param y: placeholder; has no effect whatsoever.
        :type y: `np.ndarray`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        noise = np.random.rand(*x.shape).astype(np.float32)
        noise = (2 * noise -1) * self.eps
        return x + noise
