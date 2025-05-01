import logging
import numpy as np
import pandas as pd
import itertools
import torch
from sklearn.mixture import GaussianMixture
from pomegranate.bayesian_network import BayesianNetwork
from .common import Goal, SingleTableMetric, IncomputableMetricError
LOGGER = logging.getLogger(__name__)
pd.set_option('future.no_silent_downcasting', True)

class BNLikelihoodBase(SingleTableMetric):
    """BayesianNetwork Likelihood Single Table base metric."""

    @classmethod
    def _likelihoods(cls, real_data, synthetic_data, metadata=None, structure=None):
        
        real_data, synthetic_data, metadata = cls._validate_inputs(
            real_data, synthetic_data, metadata
        )
        structure = metadata.get('structure', structure)
        fields = cls._select_fields(metadata, ('categorical', 'boolean'))

        if not fields:
            return np.full(len(real_data), np.nan)

        LOGGER.debug('Fitting the BayesianNetwork to the real data')
        bn = BayesianNetwork(structure=structure if structure else None, algorithm='chow-liu')
        category_to_integer = {
            column: {
                category: i
                for i, category in enumerate(
                    pd.concat([real_data[column], synthetic_data[column]]).unique()
                )
            }
            for column in fields
        }
        real_data[fields] = real_data[fields].replace(category_to_integer).astype('int64')
        synthetic_data[fields] = synthetic_data[fields].replace(category_to_integer).astype('int64')

        bn.fit(torch.tensor(real_data[fields].to_numpy()))
        LOGGER.debug('Evaluating likelihood of the synthetic data')
        probabilities = []
        for _, row in synthetic_data[fields].iterrows():
            try:
                probabilities.append(bn.probability([row.to_numpy()]).item())
            except ValueError:
                probabilities.append(0)

        return np.asarray(probabilities)


class BNLikelihood(BNLikelihoodBase):
    """BayesianNetwork Likelihood Single Table metric.

    This metric fits a BayesianNetwork to the real data and then evaluates how
    likely it is that the synthetic data belongs to the same distribution.

    The output is the average probability across all the synthetic rows.

    Attributes:
        name (str):
            Name to use when reports about this metric are printed.
        goal (sdmetrics.goal.Goal):
            The goal of this metric.
        min_value (Union[float, tuple[float]]):
            Minimum value or values that this metric can take.
        max_value (Union[float, tuple[float]]):
            Maximum value or values that this metric can take.
    """

    name = 'BayesianNetwork Likelihood'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None, structure=None):
        """Compute this metric.

        This fits a BayesianNetwork to the real data and then evaluates how
        likely it is that the synthetic data belongs to the same distribution.

        Real data and synthetic data must be passed as ``pandas.DataFrame`` instances
        and ``metadata`` as a ``Table`` metadata ``dict`` representation.

        If no ``metadata`` is given, one will be built from the values observed
        in the ``real_data``.

        If a ``structure`` is given, either directly or as a ``structure`` first level
        entry within the ``metadata`` dict, it is passed to the underlying BayesianNetwork
        for fitting. Otherwise, the structure is learned from the data using the ``chow-liu``
        algorithm.

        ``structure`` can be passed as either a tuple of tuples representing only the
        network structure or as a ``dict`` representing a full serialization of a previously
        fitted ``BayesianNetwork``. In the later scenario, only the ``structure`` will be
        extracted from the ``BayesianNetwork`` instance, and then a new one will be fitted
        to the given data.

        The output is the average probability across all the synthetic rows.

        Args:
            real_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the synthetic dataset.
            metadata (dict):
                Table metadata dict. If not passed, it is build based on the
                real_data fields and dtypes. Optionally, the metadata can include
                a ``structure`` entry with the structure of the Bayesian Network.
            structure (dict):
                Optional. BayesianNetwork structure to use when fitting
                to the real data. If not passed, learn it from the data
                using the ``chow-liu`` algorith. This is ignored if ``metadata``
                is passed and it contains a ``structure`` entry in it.

        Returns:
            float:
                Mean of the probabilities returned by the Bayesian Network.
        """
        return np.mean(cls._likelihoods(real_data, synthetic_data, metadata, structure))


class BNLogLikelihood(BNLikelihoodBase):
    """BayesianNetwork Log Likelihood Single Table metric.

    This metric fits a BayesianNetwork to the real data and then evaluates how
    likely it is that the synthetic data belongs to the same distribution.

    The output is the average log probability across all the synthetic rows.

    Attributes:
        name (str):
            Name to use when reports about this metric are printed.
        goal (sdmetrics.goal.Goal):
            The goal of this metric.
        min_value (Union[float, tuple[float]]):
            Minimum value or values that this metric can take.
        max_value (Union[float, tuple[float]]):
            Maximum value or values that this metric can take.
    """

    name = 'BayesianNetwork Log Likelihood'
    goal = Goal.MAXIMIZE
    min_value = -np.inf
    max_value = 0

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None, structure=None):
        """Compute this metric.

        This fits a BayesianNetwork to the real data and then evaluates how
        likely it is that the synthetic data belongs to the same distribution.

        Real data and synthetic data must be passed as ``pandas.DataFrame`` instances
        and ``metadata`` as a ``Table`` metadata ``dict`` representation.

        If no ``metadata`` is given, one will be built from the values observed
        in the ``real_data``.

        If a ``structure`` is given, either directly or as a ``structure`` first level
        entry within the ``metadata`` dict, it is passed to the underlying BayesianNetwork
        for fitting. Otherwise, the structure is learned from the data using the ``chow-liu``
        algorithm.

        ``structure`` can be passed as either a tuple of tuples representing only the
        network structure or as a ``dict`` representing a full serialization of a previously
        fitted ``BayesianNetwork``. In the later scenario, only the ``structure`` will be
        extracted from the ``BayesianNetwork`` instance, and then a new one will be fitted
        to the given data.

        The output is the average log probability across all the synthetic rows.

        Args:
            real_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the synthetic dataset.
            metadata (dict):
                Table metadata dict. If not passed, it is build based on the
                real_data fields and dtypes. Optionally, the metadata can include
                a ``structure`` entry with the structure of the Bayesian Network.
            structure (dict):
                Optional. BayesianNetwork structure to use when fitting
                to the real data. If not passed, learn it from the data
                using the ``chow-liu`` algorith. This is ignored if ``metadata``
                is passed and it contains a ``structure`` entry in it.

        Returns:
            float:
                Mean of the log probabilities returned by the Bayesian Network.
        """
        likelihoods = cls._likelihoods(real_data, synthetic_data, metadata, structure)
        likelihoods[np.where(likelihoods == 0)] = 1e-8
        return np.mean(np.log(likelihoods))

    @classmethod
    def normalize(cls, raw_score):
        """Normalize the log-likelihood value.

        Note that this is not the mean likelihood but rather the exponentiation
        of the mean log-likelihood.

        Args:
            raw_score (float):
                The value of the metric from `compute`.

        Returns:
            float:
                The normalized value of the metric
        """
        return super().normalize(raw_score)


class GMLogLikelihood(SingleTableMetric):
    """GaussianMixture Single Table metric.

    This metric fits multiple GaussianMixture models to the real data and then
    evaluates how likely it is that the synthetic data belongs to the same
    distribution as the real data.

    By default, GaussianMixture models with 10, 20 and 30 components are
    fitted a total of 3 times.

    The output is the average log likelihood across all the GMMs.

    Attributes:
        name (str):
            Name to use when reports about this metric are printed.
        goal (sdmetrics.goal.Goal):
            The goal of this metric.
        min_value (Union[float, tuple[float]]):
            Minimum value or values that this metric can take.
        max_value (Union[float, tuple[float]]):
            Maximum value or values that this metric can take.
    """

    name = 'GaussianMixture Log Likelihood'
    goal = Goal.MAXIMIZE
    min_value = -np.inf
    max_value = np.inf

    @classmethod
    def _select_gmm(cls, real_data, n_components, covariance_type):
        if isinstance(n_components, int):
            min_comp = max_comp = n_components
        else:
            min_comp, max_comp = n_components

        if isinstance(covariance_type, str):
            covariance_type = (covariance_type,)

        combinations = list(itertools.product(range(min_comp, max_comp + 1), covariance_type))
        if len(combinations) == 1:
            return combinations[0]

        lowest_bic = np.inf
        best = None
        for n_components, covariance_type in combinations:
            gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type)
            try:
                gmm.fit(real_data)
                bic = gmm.bic(real_data)
                LOGGER.debug('%s, %s: %s', n_components, covariance_type, bic)
                if bic < lowest_bic:
                    lowest_bic = bic
                    best = (n_components, covariance_type)

            except ValueError:
                pass

        if not best:
            metric_name = cls.name
            raise IncomputableMetricError(f'{metric_name}: Unable to fit GaussianMixture')

        return best

    @classmethod
    def compute(
        cls,
        real_data,
        synthetic_data,
        metadata=None,
        n_components=(1, 30),
        covariance_type='diag',
        iterations=3,
        retries=3,
    ):
        """Compute this metric.

        This fits multiple GaussianMixture models to the real data and then
        evaluates how likely it is that the synthetic data belongs to the same
        distribution as the real data.

        By default, GaussianMixture models will search for the optimal number of
        components and covariance type using the real data and then evaluate
        the likelihood of the synthetic data using those arguments 3 times.

        Real data and synthetic data must be passed as ``pandas.DataFrame`` instances
        and ``metadata`` as a ``Table`` metadata ``dict`` representation.

        If no ``metadata`` is given, one will be built from the values observed
        in the ``real_data``.

        The output is the average log likelihood across all the GMMs evaluated.

        Args:
            real_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the synthetic dataset.
            metadata (dict):
                Table metadata dict.
            n_components (Union[int, tuple[int]]):
                Number of components to use for the GMM. If a tuple with
                2 integers is passed, the optimal number of components within
                the range will be searched. Defaults to (1, 30)
            covariance_type (Union[str, tuple[str]]):
                Covariange type to use for the GMM. If multiple values are
                passed, the best one will be searched. Defaults to ``'diag'``.
            iterations (int):
                Number of times that each number of components should
                be evaluated before averaging the scores. Defaults to 3.
            retries (int):
                Number of times that each iteration will be retried if the
                GMM model crashes during fit. Defaults to 3.

        Returns:
            float:
                Average score returned by the GaussianMixtures.
        """
        real_data, synthetic_data, metadata = cls._validate_inputs(
            real_data, synthetic_data, metadata
        )
        fields = cls._select_fields(metadata, 'numerical')

        real_data = real_data[fields]
        synthetic_data = synthetic_data[fields]
        real_data = real_data.fillna(real_data.mean())
        synthetic_data = synthetic_data.fillna(synthetic_data.mean())

        if not isinstance(n_components, int) or not isinstance(covariance_type, str):
            LOGGER.debug('Selecting best GMM parameters')
            best_gmm = cls._select_gmm(real_data, n_components, covariance_type)
            if best_gmm is None:
                return np.nan

            n_components, covariance_type = best_gmm
            LOGGER.debug(
                'n_components=%s and covariance_type=%s selected', n_components, covariance_type
            )

        scores = []
        for _ in range(iterations * retries):
            try:
                gmm = GaussianMixture(n_components, covariance_type=covariance_type)
                gmm.fit(real_data)
                scores.append(gmm.score(synthetic_data))
                if len(scores) >= iterations:
                    break
            except ValueError:
                pass

        if not scores:
            metric_name = cls.name
            raise IncomputableMetricError(f'{metric_name}: Exhausted retries for GaussianMixture')

        return np.mean(scores)

    @classmethod
    def normalize(cls, raw_score):
        """Normalize the log-likelihood value.

        Notice that this is not the mean likelihood.

        Args:
            raw_score (float):
                The value of the metric from `compute`.

        Returns:
            float:
                The normalized value of the metric
        """
        return super().normalize(raw_score)