# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict
import pandas as pd
import numpy as np
from scipy.stats import logistic


class EventGenerator():
    """
    This class takes as unique input a map that indicates the relevance of a feature
    with the probability (more precisely, the logit) of an event.

    It works similar to a trained ML module, with the exception that the coefficients
    for our model are not infered from a dataset, but passed when the class' instance
    is defined
    """

    def __init__(self,
                 features_map: Dict[str,
                                    object],
                 scale=None,
                 baseline: float = 0,
                 seed: int = None) -> None:
        """
        Inputs
        - features_map: dictionary that translates the value of each feature to the outcome (e.g. how much iOS vs Android increase probability of conversion)
        - scale: parameter associated with the variance of the distribution and mean different things depending on the selected distribution.
                 For Binominal it means the number of samples per user. For log-normal distribution, it sets the variance of the final distribution
        - baseline: constant value to be added when calculating the value of each row
        - seed: defines the random-seed for pseudo-random number generator, which is used to generate the samples
        """
        self.features_map = features_map
        self.scale = scale if scale is not None else 1
        self.baseline = baseline
        self.rng = seed if isinstance(
            seed, np.random.Generator) else np.random.default_rng(seed)

    def _sample_from_distribution(
            self,
            locs: np.ndarray,
            scale: float = None) -> np.ndarray:
        """
        Generate samples based on the assumed distribution for the variable
        """
        raise NotImplementedError

    def generate_events(self, data: pd.DataFrame, scale: float = None):
        """
        Applies the mapping of features->value for each row in the dataframe and sum all contributions together.
        The result then gets passed to a sampling algorithm
        Inputs
            - data: dataframe containing the columns that serve to define the impact in the distribution
                    Ex: if country == 'US', double the probability of the event
            - scale: a float, pd.Series or np.ndarray which tweaks the distribution in consideration. For Binomial distributions.
                    it defines the number of samples per user. For log-normal distribution, it sets the variance of the final distribution
        """
        locs = data.apply(self.features_map).sum(axis=1) + self.baseline
        scale = scale if scale is not None else self.scale
        return self._sample_from_distribution(locs, scale)


class BinomialEventGenerator(EventGenerator):
    def __init__(self,
                 features_map: Dict[str,
                                    object],
                 scale=None,
                 baseline: float = 0,
                 seed: int = None,
                 logit_output: bool = True) -> None:
        super().__init__(features_map, scale, baseline, seed)
        self.logit_output = logit_output

    def _sample_from_distribution(
            self,
            locs: np.ndarray,
            scale: float = None) -> np.ndarray:
        """
        Applies the mapping of features->value for each row in the dataframe and sum all contributions together.
        Then generates samples based on the results of each row independently assuming a Binomial distribution where
        'scale' represents the number of trials
        """
        probabilities = logistic.cdf(locs) if self.logit_output else locs
        return self.rng.binomial(scale, probabilities)


class LognormalEventGenerator(EventGenerator):
    def _sample_from_distribution(
            self,
            locs: np.ndarray,
            scale: float = None) -> np.ndarray:
        """
        Applies the mapping of features->value for each row in the dataframe and sum all contributions together.
        Then generates samples based on the results of each row independently assuming a log-normal distribution where
        'scale' represents the standard deviation of the log-normal distribution
        """
        mean = np.log(locs / (
            ((scale**2) / (locs**2) + 1
             )**0.5))
        sigma = np.log((scale**2) / (locs**2) + 1)**0.5

        return self.rng.lognormal(mean, sigma)


class ParetoEventGenerator(EventGenerator):
    def _sample_from_distribution(
            self,
            locs: np.ndarray,
            scale: float = 1) -> np.ndarray:
        """
        Applies the mapping of features->value for each row in the dataframe and sum all contributions together.
        Then generates samples based on the results of each row independently assuming a Pareto distribution
        """
        alpha = locs / (locs - 1)
        return self.rng.pareto(alpha) + 1
