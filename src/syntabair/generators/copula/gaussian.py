"""GaussianMultivariate module."""
# This file includes code adapted from the Copulas project:
# https://github.com/sdv-dev/Copulas
# Licensed under the Business Source License 1.1.
# Modifications have been made to suit this project's requirements.

import logging
import sys

import numpy as np
import pandas as pd
from scipy import stats

import numpy as np
from scipy.stats import norm

from .base import BoundedType, ParametricType, ScipyModel, Univariate, Multivariate
from .utils import (
    EPSILON,
    check_valid_values,
    get_instance,
    get_qualified_name,
    random_state,
    store_args,
    validate_random_state,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_DISTRIBUTION = Univariate


class GaussianUnivariate(ScipyModel):
    """Gaussian univariate model."""

    PARAMETRIC = ParametricType.PARAMETRIC
    BOUNDED = BoundedType.UNBOUNDED

    MODEL_CLASS = norm

    def _fit_constant(self, X):
        self._params = {'loc': np.unique(X)[0], 'scale': 0}

    def _fit(self, X):
        self._params = {'loc': np.mean(X), 'scale': np.std(X)}

    def _is_constant(self):
        return self._params['scale'] == 0

    def _extract_constant(self):
        return self._params['loc']



class GaussianMultivariate(Multivariate):
    """Class for a multivariate distribution that uses the Gaussian copula.

    Args:
        distribution (str or dict):
            Fully qualified name of the class to be used for modeling the marginal
            distributions or a dictionary mapping column names to the fully qualified
            distribution names.
    """

    correlation = None
    columns = None
    univariates = None

    @store_args
    def __init__(self, distribution=DEFAULT_DISTRIBUTION, random_state=None):
        self.random_state = validate_random_state(random_state)
        self.distribution = distribution

    def __repr__(self):
        """Produce printable representation of the object."""
        if self.distribution == DEFAULT_DISTRIBUTION:
            distribution = ''
        elif isinstance(self.distribution, type):
            distribution = f'distribution="{self.distribution.__name__}"'
        else:
            distribution = f'distribution="{self.distribution}"'

        return f'GaussianMultivariate({distribution})'

    def _transform_to_normal(self, X):
        if isinstance(X, pd.Series):
            X = X.to_frame().T
        elif not isinstance(X, pd.DataFrame):
            if len(X.shape) == 1:
                X = [X]

            X = pd.DataFrame(X, columns=self.columns)

        U = []
        for column_name, univariate in zip(self.columns, self.univariates):
            if column_name in X:
                column = X[column_name]
                U.append(univariate.cdf(column.to_numpy()).clip(EPSILON, 1 - EPSILON))

        return stats.norm.ppf(np.column_stack(U))

    @check_valid_values
    def fit(self, X):
        """Compute the distribution for each variable and then its correlation matrix.

        Arguments:
            X (pandas.DataFrame):
                Values of the random variables.
        """
        LOGGER.info('Fitting %s', self)

        # Validate the input data
        X = self._validate_input(X)
        columns, univariates = self._fit_columns(X)

        self.columns = columns
        self.univariates = univariates

        LOGGER.debug('Computing correlation.')
        self.correlation = self._get_correlation(X)
        self.fitted = True
        LOGGER.debug('GaussianMultivariate fitted successfully')

    def _validate_input(self, X):
        """Validate the input data."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        return X

    def _fit_columns(self, X):
        """Fit each column to its distribution."""
        columns = []
        univariates = []
        for column_name, column in X.items():
            distribution = self._get_distribution_for_column(column_name)
            LOGGER.debug('Fitting column %s to %s', column_name, distribution)

            univariate = self._fit_column(column, distribution, column_name)
            columns.append(column_name)
            univariates.append(univariate)

        return columns, univariates

    def _get_distribution_for_column(self, column_name):
        """Retrieve the distribution for a given column name."""
        if isinstance(self.distribution, dict):
            return self.distribution.get(column_name, DEFAULT_DISTRIBUTION)

        return self.distribution

    def _fit_column(self, column, distribution, column_name):
        """Fit a single column to its distribution with exception handling."""
        univariate = get_instance(distribution)
        try:
            univariate.fit(column)
        except Exception as error:
            univariate = self._fit_with_fallback_distribution(
                column, distribution, column_name, error
            )

        return univariate

    def _fit_with_fallback_distribution(self, column, distribution, column_name, error):
        """Fall back to fitting a Gaussian distribution and log the error."""
        log_message = (
            f'Unable to fit to a {distribution} distribution for column {column_name}. '
            'Using a Gaussian distribution instead.'
        )
        LOGGER.info(log_message)
        univariate = GaussianUnivariate()
        univariate.fit(column)
        return univariate

    def _get_correlation(self, X):
        """Compute correlation matrix with transformed data.

        Args:
            X (numpy.ndarray):
                Data for which the correlation needs to be computed.

        Returns:
            numpy.ndarray:
                computed correlation matrix.
        """
        result = self._transform_to_normal(X)
        correlation = pd.DataFrame(data=result).corr().to_numpy()
        correlation = np.nan_to_num(correlation, nan=0.0)
        # If singular, add some noise to the diagonal
        if np.linalg.cond(correlation) > 1.0 / sys.float_info.epsilon:
            correlation = correlation + np.identity(correlation.shape[0]) * EPSILON

        return pd.DataFrame(correlation, index=self.columns, columns=self.columns)

    def probability_density(self, X):
        """Compute the probability density for each point in X.

        Arguments:
            X (pandas.DataFrame):
                Values for which the probability density will be computed.

        Returns:
            numpy.ndarray:
                Probability density values for points in X.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()
        transformed = self._transform_to_normal(X)

        return stats.multivariate_normal.pdf(transformed, cov=self.correlation, allow_singular=True)

    def cumulative_distribution(self, X):
        """Compute the cumulative distribution value for each point in X.

        Arguments:
            X (pandas.DataFrame):
                Values for which the cumulative distribution will be computed.

        Returns:
            numpy.ndarray:
                Cumulative distribution values for points in X.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()
        transformed = self._transform_to_normal(X)
        return stats.multivariate_normal.cdf(transformed, cov=self.correlation)

    def _get_conditional_distribution(self, conditions):
        """Compute the parameters of a conditional multivariate normal distribution.

        The parameters of the conditioned distribution are computed as specified here:
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions

        Args:
            conditions (pandas.Series):
                Mapping of the column names and column values to condition on.
                The input values have already been transformed to their normal distribution.

        Returns:
            tuple:
                * means (numpy.array):
                    mean values to use for the conditioned multivariate normal.
                * covariance (numpy.array):
                    covariance matrix to use for the conditioned
                  multivariate normal.
                * columns (list):
                    names of the columns that will be sampled conditionally.
        """
        columns2 = conditions.index
        columns1 = self.correlation.columns.difference(columns2)

        sigma11 = self.correlation.loc[columns1, columns1].to_numpy()
        sigma12 = self.correlation.loc[columns1, columns2].to_numpy()
        sigma21 = self.correlation.loc[columns2, columns1].to_numpy()
        sigma22 = self.correlation.loc[columns2, columns2].to_numpy()

        mu1 = np.zeros(len(columns1))
        mu2 = np.zeros(len(columns2))

        sigma12sigma22inv = sigma12 @ np.linalg.inv(sigma22)

        mu_bar = mu1 + sigma12sigma22inv @ (conditions - mu2)
        sigma_bar = sigma11 - sigma12sigma22inv @ sigma21

        return mu_bar, sigma_bar, columns1

    def _get_normal_samples(self, num_rows, conditions):
        """Get random rows in the standard normal space.

        If no conditions are given, the values are sampled from a standard normal
        multivariate.

        If conditions are given, they are transformed to their equivalent standard
        normal values using their marginals and then the values are sampled from
        a standard normal multivariate conditioned on the given condition values.
        """
        if conditions is None:
            covariance = self.correlation
            columns = self.columns
            means = np.zeros(len(columns))
        else:
            conditions = pd.Series(conditions)
            normal_conditions = self._transform_to_normal(conditions)[0]
            normal_conditions = pd.Series(normal_conditions, index=conditions.index)
            means, covariance, columns = self._get_conditional_distribution(normal_conditions)

        samples = np.random.multivariate_normal(means, covariance, size=num_rows)
        return pd.DataFrame(samples, columns=columns)

    @random_state
    def sample(self, num_rows=1, conditions=None):
        """Sample values from this model.

        Argument:
            num_rows (int):
                Number of rows to sample.
            conditions (dict or pd.Series):
                Mapping of the column names and column values to condition on.

        Returns:
            numpy.ndarray:
                Array of shape (n_samples, *) with values randomly
                sampled from this model distribution. If conditions have been
                given, the output array also contains the corresponding columns
                populated with the given values.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()

        samples = self._get_normal_samples(num_rows, conditions)

        output = {}
        for column_name, univariate in zip(self.columns, self.univariates):
            if conditions and column_name in conditions:
                # Use the values that were given as conditions in the original space.
                output[column_name] = np.full(num_rows, conditions[column_name])
            else:
                cdf = stats.norm.cdf(samples[column_name])
                output[column_name] = univariate.percent_point(cdf)

        return pd.DataFrame(data=output)

    def to_dict(self):
        """Return a `dict` with the parameters to replicate this object.

        Returns:
            dict:
                Parameters of this distribution.
        """
        self.check_fit()
        univariates = [univariate.to_dict() for univariate in self.univariates]

        return {
            'correlation': self.correlation.to_numpy().tolist(),
            'univariates': univariates,
            'columns': self.columns,
            'type': get_qualified_name(self),
        }

    @classmethod
    def from_dict(cls, copula_dict):
        """Create a new instance from a parameters dictionary.

        Args:
            params (dict):
                Parameters of the distribution, in the same format as the one
                returned by the ``to_dict`` method.

        Returns:
            Multivariate:
                Instance of the distribution defined on the parameters.
        """
        instance = cls()
        instance.univariates = []
        columns = copula_dict['columns']
        instance.columns = columns

        for parameters in copula_dict['univariates']:
            instance.univariates.append(Univariate.from_dict(parameters))

        correlation = copula_dict['correlation']
        instance.correlation = pd.DataFrame(correlation, index=columns, columns=columns)
        instance.fitted = True

        return instance