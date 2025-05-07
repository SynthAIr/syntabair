"""Correlation preservation metrics for comparing synthetic data with real data.

This module implements metrics that measure how well the synthetic data preserves
the correlation structures found in the real data. It includes metrics for comparing
Pearson correlation, Spearman rank correlation, and correlation matrix distances.
"""
# This file includes code adapted from the SDMetrics project:
# https://github.com/sdv-dev/SDMetrics
# Licensed under the MIT License.
# Modifications have been made to suit this project's requirements.

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.preprocessing import StandardScaler

from .common import Goal, ColumnPairsMetric, MultiColumnPairsMetric, SingleTableMetric


class ColumnPairsCorrelationSimilarity(ColumnPairsMetric):
    """Base class for correlation similarity metrics.

    This provides common functionality for evaluating how well the correlation
    between two columns is preserved in the synthetic data.
    """

    name = 'Column Correlation Similarity'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @staticmethod
    def _compute_correlation(x, y, method):
        """Compute correlation coefficient between two columns using specified method.

        Args:
            x (pandas.Series): First column values
            y (pandas.Series): Second column values
            method (str): Correlation method: 'pearson', 'spearman', or 'kendall'

        Returns:
            float: Correlation coefficient
        """
        # Drop rows where either value is NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]
        
        # If not enough data points, return NaN
        if len(x_clean) < 2:
            return np.nan
            
        if method == 'pearson':
            corr, _ = pearsonr(x_clean, y_clean)
        elif method == 'spearman':
            corr, _ = spearmanr(x_clean, y_clean)
        elif method == 'kendall':
            corr, _ = kendalltau(x_clean, y_clean)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        return corr

    @staticmethod
    def _compute_similarity(real_corr, synth_corr):
        """Compute similarity between real and synthetic correlation values.

        Args:
            real_corr (float): Correlation in the real data
            synth_corr (float): Correlation in the synthetic data

        Returns:
            float: Similarity score between 0 and 1
        """
        # Handle NaN values
        if np.isnan(real_corr) or np.isnan(synth_corr):
            return np.nan
            
        # Calculate absolute difference, scaled to [0, 1]
        # This gives 1.0 for perfect correlation preservation and 0.0 for maximum difference
        return 1.0 - min(abs(real_corr - synth_corr), 2.0) / 2.0

    @classmethod
    def compute_correlation_similarity(cls, real_data, synthetic_data, method):
        """Compute correlation similarity for a pair of columns.

        Args:
            real_data (pandas.DataFrame):
                The values from the real dataset, with 2 columns.
            synthetic_data (pandas.DataFrame):
                The values from the synthetic dataset, with 2 columns.
            method (str):
                Correlation method: 'pearson', 'spearman', or 'kendall'

        Returns:
            float:
                Correlation similarity score between 0 and 1.
        """
        columns = real_data.columns[:2]
        
        # Compute correlations
        real_corr = cls._compute_correlation(
            real_data[columns[0]], real_data[columns[1]], method
        )
        synth_corr = cls._compute_correlation(
            synthetic_data[columns[0]], synthetic_data[columns[1]], method
        )
        
        # Compute similarity score
        similarity = cls._compute_similarity(real_corr, synth_corr)
        
        return similarity


class PearsonCorrelationSimilarity(ColumnPairsCorrelationSimilarity):
    """Pearson correlation similarity metric.

    This metric measures how well the linear correlation between two columns
    is preserved in the synthetic data. It returns a score between 0 and 1,
    where 1 indicates perfect preservation of the correlation and 0 indicates
    maximum deviation.

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

    name = 'Pearson Correlation Similarity'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @staticmethod
    def compute(real_data, synthetic_data):
        """Compute the Pearson correlation similarity for a pair of columns.

        Args:
            real_data (pandas.DataFrame):
                The values from the real dataset, with 2 columns.
            synthetic_data (pandas.DataFrame):
                The values from the synthetic dataset, with 2 columns.

        Returns:
            float:
                Pearson correlation similarity score between 0 and 1.
        """
        return ColumnPairsCorrelationSimilarity.compute_correlation_similarity(
            real_data, synthetic_data, 'pearson'
        )

    @classmethod
    def compute_breakdown(cls, real_data, synthetic_data):
        """Compute the breakdown of this metric.

        Args:
            real_data (pandas.DataFrame):
                The values from the real dataset, with 2 columns.
            synthetic_data (pandas.DataFrame):
                The values from the synthetic dataset, with 2 columns.

        Returns:
            dict:
                A dictionary containing the score as well as the individual correlation values.
        """
        columns = real_data.columns[:2]
        
        # Compute correlations
        real_corr = cls._compute_correlation(
            real_data[columns[0]], real_data[columns[1]], 'pearson'
        )
        synth_corr = cls._compute_correlation(
            synthetic_data[columns[0]], synthetic_data[columns[1]], 'pearson'
        )
        
        # Compute similarity score
        similarity = cls._compute_similarity(real_corr, synth_corr)
        
        return {
            'score': similarity,
            'real_correlation': real_corr,
            'synthetic_correlation': synth_corr,
        }


class SpearmanCorrelationSimilarity(ColumnPairsCorrelationSimilarity):
    """Spearman rank correlation similarity metric.

    This metric measures how well the monotonic relationship between two columns
    is preserved in the synthetic data. It returns a score between 0 and 1,
    where 1 indicates perfect preservation of the rank correlation and 0 indicates
    maximum deviation.

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

    name = 'Spearman Correlation Similarity'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @staticmethod
    def compute(real_data, synthetic_data):
        """Compute the Spearman correlation similarity for a pair of columns.

        Args:
            real_data (pandas.DataFrame):
                The values from the real dataset, with 2 columns.
            synthetic_data (pandas.DataFrame):
                The values from the synthetic dataset, with 2 columns.

        Returns:
            float:
                Spearman correlation similarity score between 0 and 1.
        """
        return ColumnPairsCorrelationSimilarity.compute_correlation_similarity(
            real_data, synthetic_data, 'spearman'
        )

    @classmethod
    def compute_breakdown(cls, real_data, synthetic_data):
        """Compute the breakdown of this metric.

        Args:
            real_data (pandas.DataFrame):
                The values from the real dataset, with 2 columns.
            synthetic_data (pandas.DataFrame):
                The values from the synthetic dataset, with 2 columns.

        Returns:
            dict:
                A dictionary containing the score as well as the individual correlation values.
        """
        columns = real_data.columns[:2]
        
        # Compute correlations
        real_corr = cls._compute_correlation(
            real_data[columns[0]], real_data[columns[1]], 'spearman'
        )
        synth_corr = cls._compute_correlation(
            synthetic_data[columns[0]], synthetic_data[columns[1]], 'spearman'
        )
        
        # Compute similarity score
        similarity = cls._compute_similarity(real_corr, synth_corr)
        
        return {
            'score': similarity,
            'real_correlation': real_corr,
            'synthetic_correlation': synth_corr,
        }


class KendallCorrelationSimilarity(ColumnPairsCorrelationSimilarity):
    """Kendall's Tau rank correlation similarity metric.

    This metric measures how well the concordance between pairs of observations
    is preserved in the synthetic data. It returns a score between 0 and 1,
    where 1 indicates perfect preservation of the Kendall's Tau and 0 indicates
    maximum deviation.

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

    name = 'Kendall Correlation Similarity'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @staticmethod
    def compute(real_data, synthetic_data):
        """Compute the Kendall correlation similarity for a pair of columns.

        Args:
            real_data (pandas.DataFrame):
                The values from the real dataset, with 2 columns.
            synthetic_data (pandas.DataFrame):
                The values from the synthetic dataset, with 2 columns.

        Returns:
            float:
                Kendall correlation similarity score between 0 and 1.
        """
        return ColumnPairsCorrelationSimilarity.compute_correlation_similarity(
            real_data, synthetic_data, 'kendall'
        )

    @classmethod
    def compute_breakdown(cls, real_data, synthetic_data):
        """Compute the breakdown of this metric.

        Args:
            real_data (pandas.DataFrame):
                The values from the real dataset, with 2 columns.
            synthetic_data (pandas.DataFrame):
                The values from the synthetic dataset, with 2 columns.

        Returns:
            dict:
                A dictionary containing the score as well as the individual correlation values.
        """
        columns = real_data.columns[:2]
        
        # Compute correlations
        real_corr = cls._compute_correlation(
            real_data[columns[0]], real_data[columns[1]], 'kendall'
        )
        synth_corr = cls._compute_correlation(
            synthetic_data[columns[0]], synthetic_data[columns[1]], 'kendall'
        )
        
        # Compute similarity score
        similarity = cls._compute_similarity(real_corr, synth_corr)
        
        return {
            'score': similarity,
            'real_correlation': real_corr,
            'synthetic_correlation': synth_corr,
        }


class CorrelationMatrixDistance(SingleTableMetric):
    """Correlation matrix distance metric.

    This metric computes the Frobenius norm of the difference between the
    correlation matrices of the real and synthetic datasets, normalized to
    a score between 0 and 1. A score of 1 indicates identical correlation
    matrices, while a score of 0 indicates maximum difference.

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

    name = 'Correlation Matrix Distance'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @staticmethod
    def _compute_correlation_matrix(data, method='pearson'):
        """Compute the correlation matrix for a dataset.

        Args:
            data (pandas.DataFrame):
                The dataset to compute correlations for.
            method (str):
                Correlation method: 'pearson', 'spearman', or 'kendall'.

        Returns:
            pandas.DataFrame:
                The correlation matrix.
        """
        # Replace any remaining NaN values with the column mean to avoid correlation issues
        data_filled = data.copy()
        for col in data_filled.columns:
            if data_filled[col].isna().any():
                data_filled[col] = data_filled[col].fillna(data_filled[col].mean())
        
        return data_filled.corr(method=method)

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None, method='pearson'):
        """Compute the correlation matrix distance metric.

        Args:
            real_data (pandas.DataFrame):
                The values from the real dataset.
            synthetic_data (pandas.DataFrame):
                The values from the synthetic dataset.
            metadata (dict):
                Table metadata dict.
            method (str):
                Correlation method: 'pearson', 'spearman', or 'kendall'.

        Returns:
            float:
                Correlation matrix similarity score between 0 and 1.
        """
        real_data, synthetic_data, metadata = cls._validate_inputs(
            real_data, synthetic_data, metadata
        )
        numerical_columns = cls._select_fields(metadata, 'numerical')
        
        if len(numerical_columns) < 2:
            # Not enough numerical columns for correlation
            return np.nan
        
        # Compute correlation matrices
        real_corr = cls._compute_correlation_matrix(real_data[numerical_columns], method)
        synth_corr = cls._compute_correlation_matrix(synthetic_data[numerical_columns], method)
        
        # Compute the Frobenius norm of the difference
        diff_norm = np.linalg.norm(real_corr - synth_corr, 'fro')
        
        # Normalize to [0, 1] range, where 1 is perfect correlation preservation
        # The maximum possible Frobenius norm for difference of correlation matrices
        # is sqrt(2 * n * (n-1)) where n is the number of variables
        n = len(numerical_columns)
        max_norm = np.sqrt(2 * n * (n - 1))
        
        # Handle the case when max_norm is 0 (e.g., when n=1)
        if max_norm == 0:
            return 1.0
            
        similarity = 1.0 - (diff_norm / max_norm)
        
        # Ensure the score is in [0, 1] range
        return max(0.0, min(1.0, similarity))

    @classmethod
    def compute_breakdown(cls, real_data, synthetic_data, metadata=None, method='pearson'):
        """Compute the breakdown of this metric.

        Args:
            real_data (pandas.DataFrame):
                The values from the real dataset.
            synthetic_data (pandas.DataFrame):
                The values from the synthetic dataset.
            metadata (dict):
                Table metadata dict.
            method (str):
                Correlation method: 'pearson', 'spearman', or 'kendall'.

        Returns:
            dict:
                A dictionary containing the overall score as well as individual column pair scores.
        """
        real_data, synthetic_data, metadata = cls._validate_inputs(
            real_data, synthetic_data, metadata
        )
        numerical_columns = cls._select_fields(metadata, 'numerical')
        
        if len(numerical_columns) < 2:
            # Not enough numerical columns for correlation
            return {'score': np.nan}
        
        # Compute correlation matrices
        real_corr = cls._compute_correlation_matrix(real_data[numerical_columns], method)
        synth_corr = cls._compute_correlation_matrix(synthetic_data[numerical_columns], method)
        
        # Compute the Frobenius norm of the difference
        diff_norm = np.linalg.norm(real_corr - synth_corr, 'fro')
        
        # Normalize to [0, 1] range, where 1 is perfect correlation preservation
        n = len(numerical_columns)
        max_norm = np.sqrt(2 * n * (n - 1))
        
        # Handle the case when max_norm is 0 (e.g., when n=1)
        if max_norm == 0:
            return {'score': 1.0}
            
        similarity = 1.0 - (diff_norm / max_norm)
        
        # Ensure the score is in [0, 1] range
        similarity = max(0.0, min(1.0, similarity))
        
        # Compute pairwise correlation differences
        pairs = {}
        for i, col1 in enumerate(numerical_columns):
            for j, col2 in enumerate(numerical_columns):
                if i < j:  # Only consider upper triangle of correlation matrix
                    diff = abs(real_corr.loc[col1, col2] - synth_corr.loc[col1, col2])
                    pairs[f"{col1}_{col2}"] = 1.0 - min(diff, 2.0) / 2.0
        
        result = {'score': similarity}
        result.update(pairs)
        
        return result


class PearsonCorrelation(MultiColumnPairsMetric):
    """MultiColumnPairsMetric based on PearsonCorrelationSimilarity.

    This metric applies the PearsonCorrelationSimilarity metric to all pairs
    of numerical columns in the dataset and returns the average similarity.

    Attributes:
        name (str):
            Name to use when reports about this metric are printed.
        goal (sdmetrics.goal.Goal):
            The goal of this metric.
        min_value (Union[float, tuple[float]]):
            Minimum value or values that this metric can take.
        max_value (Union[float, tuple[float]]):
            Maximum value or values that this metric can take.
        column_pairs_metric (sdmetrics.column_pairs.base.ColumnPairsMetric):
            ColumnPairs PearsonCorrelationSimilarity.
        field_types (dict):
            Field types to which the metric will be applied.
    """

    field_types = ('numerical',)
    column_pairs_metric = PearsonCorrelationSimilarity


class SpearmanCorrelation(MultiColumnPairsMetric):
    """MultiColumnPairsMetric based on SpearmanCorrelationSimilarity.

    This metric applies the SpearmanCorrelationSimilarity metric to all pairs
    of numerical columns in the dataset and returns the average similarity.

    Attributes:
        name (str):
            Name to use when reports about this metric are printed.
        goal (sdmetrics.goal.Goal):
            The goal of this metric.
        min_value (Union[float, tuple[float]]):
            Minimum value or values that this metric can take.
        max_value (Union[float, tuple[float]]):
            Maximum value or values that this metric can take.
        column_pairs_metric (sdmetrics.column_pairs.base.ColumnPairsMetric):
            ColumnPairs SpearmanCorrelationSimilarity.
        field_types (dict):
            Field types to which the metric will be applied.
    """

    field_types = ('numerical',)
    column_pairs_metric = SpearmanCorrelationSimilarity


class KendallCorrelation(MultiColumnPairsMetric):
    """MultiColumnPairsMetric based on KendallCorrelationSimilarity.

    This metric applies the KendallCorrelationSimilarity metric to all pairs
    of numerical columns in the dataset and returns the average similarity.

    Attributes:
        name (str):
            Name to use when reports about this metric are printed.
        goal (sdmetrics.goal.Goal):
            The goal of this metric.
        min_value (Union[float, tuple[float]]):
            Minimum value or values that this metric can take.
        max_value (Union[float, tuple[float]]):
            Maximum value or values that this metric can take.
        column_pairs_metric (sdmetrics.column_pairs.base.ColumnPairsMetric):
            ColumnPairs KendallCorrelationSimilarity.
        field_types (dict):
            Field types to which the metric will be applied.
    """

    field_types = ('numerical',)
    column_pairs_metric = KendallCorrelationSimilarity


class MixedTypeCorrelation(SingleTableMetric):
    """Mixed-type correlation metric for tables with both numeric and categorical data.

    This metric measures correlation preservation for mixed data types by:
    1. For numerical pairs, using Pearson correlation
    2. For categorical pairs, using Cramer's V correlation
    3. For mixed pairs, using correlation ratio

    The result is an overall score of how well the synthetic data preserves
    correlations between all column pairs.

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

    name = 'Mixed-Type Correlation'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @staticmethod
    def _cramer_v(x, y):
        """Calculate Cramer's V statistic for categorical-categorical correlation.

        Args:
            x (pd.Series): First categorical column
            y (pd.Series): Second categorical column
            
        Returns:
            float: Cramer's V correlation value
        """
        confusion_matrix = pd.crosstab(x, y)
        n = confusion_matrix.sum().sum()
        
        # Calculate expected frequencies
        row_sums = confusion_matrix.sum(axis=1)
        col_sums = confusion_matrix.sum(axis=0)
        expected = np.outer(row_sums, col_sums) / n
        
        # Calculate chi-squared statistic
        chi2 = ((confusion_matrix - expected) ** 2 / expected).sum().sum()
        
        # Calculate Cramer's V
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

    @staticmethod
    def _correlation_ratio(categorical, numerical):
        """Calculate the correlation ratio for categorical-numerical correlation.

        Args:
            categorical (pd.Series): Categorical column
            numerical (pd.Series): Numerical column
            
        Returns:
            float: Correlation ratio value
        """
        categories = categorical.unique()
        
        # Get means and std for each category
        cat_means = np.array([numerical[categorical == cat].mean() for cat in categories])
        
        # Calculate the overall mean and variance
        overall_mean = numerical.mean()
        overall_var = numerical.var(ddof=0)
        
        # Calculate the between-group variance
        between_var = np.sum([(numerical[categorical == cat].size / numerical.size) * 
                             (cat_mean - overall_mean)**2 
                             for cat, cat_mean in zip(categories, cat_means)])
        
        # If overall variance is 0, return 0
        if overall_var == 0:
            return 0
            
        # Calculate correlation ratio
        return np.sqrt(between_var / overall_var)

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None):
        """Compute the mixed-type correlation metric.

        Args:
            real_data (pandas.DataFrame):
                The values from the real dataset.
            synthetic_data (pandas.DataFrame):
                The values from the synthetic dataset.
            metadata (dict):
                Table metadata dict.

        Returns:
            float:
                Mixed-type correlation similarity score between 0 and 1.
        """
        real_data, synthetic_data, metadata = cls._validate_inputs(
            real_data, synthetic_data, metadata
        )
        
        # Get column types
        numerical_columns = cls._select_fields(metadata, 'numerical')
        categorical_columns = cls._select_fields(metadata, ('categorical', 'boolean'))
        
        if len(numerical_columns) + len(categorical_columns) < 2:
            # Not enough columns for correlation
            return np.nan
        
        # Calculate all pairwise correlations and their differences
        correlation_diffs = []
        
        # Numerical-Numerical pairs (Pearson)
        for i, col1 in enumerate(numerical_columns):
            for j, col2 in enumerate(numerical_columns):
                if i < j:  # Only consider upper triangle
                    real_corr = real_data[[col1, col2]].corr().iloc[0, 1]
                    synth_corr = synthetic_data[[col1, col2]].corr().iloc[0, 1]
                    
                    # Handle NaN values
                    if np.isnan(real_corr) or np.isnan(synth_corr):
                        continue
                        
                    diff = abs(real_corr - synth_corr)
                    # Normalize to [0, 1], where 1 is perfect preservation
                    correlation_diffs.append(1.0 - min(diff, 2.0) / 2.0)
        
        # Categorical-Categorical pairs (Cramer's V)
        for i, col1 in enumerate(categorical_columns):
            for j, col2 in enumerate(categorical_columns):
                if i < j:  # Only consider upper triangle
                    try:
                        real_corr = cls._cramer_v(real_data[col1], real_data[col2])
                        synth_corr = cls._cramer_v(synthetic_data[col1], synthetic_data[col2])
                        
                        # Handle NaN values
                        if np.isnan(real_corr) or np.isnan(synth_corr):
                            continue
                            
                        diff = abs(real_corr - synth_corr)
                        # Normalize to [0, 1], where 1 is perfect preservation
                        correlation_diffs.append(1.0 - min(diff, 1.0))
                    except (ValueError, ZeroDivisionError):
                        # Skip problematic pairs
                        continue
        
        # Numerical-Categorical pairs (Correlation ratio)
        for num_col in numerical_columns:
            for cat_col in categorical_columns:
                try:
                    real_corr = cls._correlation_ratio(real_data[cat_col], real_data[num_col])
                    synth_corr = cls._correlation_ratio(synthetic_data[cat_col], synthetic_data[num_col])
                    
                    # Handle NaN values
                    if np.isnan(real_corr) or np.isnan(synth_corr):
                        continue
                        
                    diff = abs(real_corr - synth_corr)
                    # Normalize to [0, 1], where 1 is perfect preservation
                    correlation_diffs.append(1.0 - min(diff, 1.0))
                except (ValueError, ZeroDivisionError):
                    # Skip problematic pairs
                    continue
        
        # If no valid correlation pairs were found, return NaN
        if not correlation_diffs:
            return np.nan
            
        # Return the average correlation preservation score
        return np.mean(correlation_diffs)

    @classmethod
    def compute_breakdown(cls, real_data, synthetic_data, metadata=None):
        """Compute the breakdown of this metric.

        Args:
            real_data (pandas.DataFrame):
                The values from the real dataset.
            synthetic_data (pandas.DataFrame):
                The values from the synthetic dataset.
            metadata (dict):
                Table metadata dict.

        Returns:
            dict:
                A dictionary containing the overall score and scores by correlation type.
        """
        real_data, synthetic_data, metadata = cls._validate_inputs(
            real_data, synthetic_data, metadata
        )
        
        # Get column types
        numerical_columns = cls._select_fields(metadata, 'numerical')
        categorical_columns = cls._select_fields(metadata, ('categorical', 'boolean'))
        
        if len(numerical_columns) + len(categorical_columns) < 2:
            # Not enough columns for correlation
            return {'score': np.nan}
        
        # Calculate all pairwise correlations and their differences
        num_num_diffs = []
        cat_cat_diffs = []
        num_cat_diffs = []
        
        # Numerical-Numerical pairs (Pearson)
        for i, col1 in enumerate(numerical_columns):
            for j, col2 in enumerate(numerical_columns):
                if i < j:  # Only consider upper triangle
                    real_corr = real_data[[col1, col2]].corr().iloc[0, 1]
                    synth_corr = synthetic_data[[col1, col2]].corr().iloc[0, 1]
                    
                    # Handle NaN values
                    if np.isnan(real_corr) or np.isnan(synth_corr):
                        continue
                        
                    diff = abs(real_corr - synth_corr)
                    # Normalize to [0, 1], where 1 is perfect preservation
                    num_num_diffs.append(1.0 - min(diff, 2.0) / 2.0)
        
        # Categorical-Categorical pairs (Cramer's V)
        for i, col1 in enumerate(categorical_columns):
            for j, col2 in enumerate(categorical_columns):
                if i < j:  # Only consider upper triangle
                    try:
                        real_corr = cls._cramer_v(real_data[col1], real_data[col2])
                        synth_corr = cls._cramer_v(synthetic_data[col1], synthetic_data[col2])
                        
                        # Handle NaN values
                        if np.isnan(real_corr) or np.isnan(synth_corr):
                            continue
                            
                        diff = abs(real_corr - synth_corr)
                        # Normalize to [0, 1], where 1 is perfect preservation
                        cat_cat_diffs.append(1.0 - min(diff, 1.0))
                    except (ValueError, ZeroDivisionError):
                        # Skip problematic pairs
                        continue
        
        # Numerical-Categorical pairs (Correlation ratio)
        for num_col in numerical_columns:
            for cat_col in categorical_columns:
                try:
                    real_corr = cls._correlation_ratio(real_data[cat_col], real_data[num_col])
                    synth_corr = cls._correlation_ratio(synthetic_data[cat_col], synthetic_data[num_col])
                    
                    # Handle NaN values
                    if np.isnan(real_corr) or np.isnan(synth_corr):
                        continue
                        
                    diff = abs(real_corr - synth_corr)
                    # Normalize to [0, 1], where 1 is perfect preservation
                    num_cat_diffs.append(1.0 - min(diff, 1.0))
                except (ValueError, ZeroDivisionError):
                    # Skip problematic pairs
                    continue
        
        # Calculate scores by type
        num_num_score = np.mean(num_num_diffs) if num_num_diffs else np.nan
        cat_cat_score = np.mean(cat_cat_diffs) if cat_cat_diffs else np.nan
        num_cat_score = np.mean(num_cat_diffs) if num_cat_diffs else np.nan
        
        # Calculate overall score
        all_diffs = num_num_diffs + cat_cat_diffs + num_cat_diffs
        overall_score = np.mean(all_diffs) if all_diffs else np.nan
        
        return {
            'score': overall_score,
            'numerical_numerical_score': num_num_score,
            'categorical_categorical_score': cat_cat_score,
            'numerical_categorical_score': num_cat_score,
        }