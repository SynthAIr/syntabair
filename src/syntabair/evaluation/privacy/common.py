
import numpy as np
import numpy as np
from copulas.univariate.base import Univariate
import copy
from operator import attrgetter
import numpy as np
import pandas as pd
from enum import Enum
from datetime import datetime
from collections import Counter
from itertools import combinations
import warnings


def majority(samples, ignore_none=True):
    """Find the most frequent element in a list.

    Arguments:
        samples (list):
            Input list. Its elements must be hashable.
        ignore_none (bool):
            If `None` is a valid value.

    Returns:
        object:
            The most frequent element in samples. Returns none if the input list is empty.
    """
    freq_dict = {}
    most_freq_ele = None
    highest_freq = 0
    for element in samples:
        if ignore_none and element is None:
            continue
        if element not in freq_dict:
            freq_dict[element] = 0

        freq = freq_dict[element] + 1
        freq_dict[element] = freq
        if freq > highest_freq:
            highest_freq = freq
            most_freq_ele = element

    return most_freq_ele


def count_frequency(samples, target):
    """Calculate how frequent an target attribute appear in a list.

    Arguments:
        samples (list):
            Input list. Its elements must be hashable.
        target (object):
            The target element.

    Returns:
        float:
            The frequency that target appears in samples. Must be in between 0 and 1.
    """
    count = 0
    for ele in samples:
        if ele == target:
            count += 1

    return count / len(samples)


def hamming_distance(target, test):
    """Calculate the hamming distance between two tuples.

    Arguments:
        target (tuple):
            The target tuple.
        test (tuple):
            The test tuple. Must have same length as target

    Returns:
        int:
            The hamming distance
    """
    dist = 0
    assert len(target) == len(test), (
        'Tuples must have the same length in the calculation of hamming distance!'
    )

    for target_entry, test_entry in zip(target, test):
        if target_entry != test_entry:
            dist += 1

    return dist


def closest_neighbors(samples, target):
    """Find elements in a given list that are closest to a given element in hamming distance.

    Arguments:
        samples (iterable[tuple]):
            The given list to look up for.
        target (tuple):
            The target tuple.

    Returns:
        list [tuple]:
            Elements in samples that are closest to target.
    """
    dist = float('inf')
    ret = []
    for element in samples:
        hamming_dist = hamming_distance(target, element)
        if hamming_dist < dist:
            dist = hamming_dist
            ret = [
                element,
            ]
        elif hamming_dist == dist:
            ret.append(element)

    return ret


def allow_nan(df):
    """Replace all invalid (`nan` and `None`) entries in a dataframe with valid placeholders.

    Arguments:
        df (pandas.DataFrame):
            The target dataframe.

    Returns:
        pandas.DataFrame:
            A modified dataframe.
    """
    df_copy = df.copy()
    for i in df_copy:
        for j in range(len(df_copy[i])):
            entry = df_copy[i][j]
            if (isinstance(entry, float) and np.isnan(entry)) or entry is None:
                df_copy[i][j] = 'place_holder_for_nan'

    return df_copy


def allow_nan_array(attributes):
    """Replace all invalid (`nan` and `None`) entries in an array with valid placeholders.

    Arguments:
        attributes (tuple):
            The target array.

    Returns:
        list:
            The modified array.
    """
    ret = []
    for entry in attributes:
        if (isinstance(entry, float) and np.isnan(entry)) or entry is None:
            ret.append('place_holder_for_nan')
        else:
            ret.append(entry)

    return ret


def validate_num_samples_num_iteration(num_rows_subsample, num_iterations):
    if num_rows_subsample is not None:
        if not isinstance(num_rows_subsample, int) or num_rows_subsample < 1:
            raise ValueError(
                f'num_rows_subsample ({num_rows_subsample}) must be an integer greater than 1.'
            )

    elif num_rows_subsample is None and num_iterations > 1:
        raise ValueError('num_iterations should not be greater than 1 if there is no subsampling.')

    if not isinstance(num_iterations, int) or num_iterations < 1:
        raise ValueError(f'num_iterations ({num_iterations}) must be an integer greater than 1.')
    
class LossFunction:
    """Base class for a loss function."""

    def fit(self, data, cols):
        """Learn the metric on the value space.

        Args:
            real_data (pandas.DataFrame):
                The real data table.
            cols (list[str]):
                The names for the target columns (usually the sensitive cols).
        """

    def measure(self, pred, real):
        """Calculate the loss of a single prediction.

        Args:
            pred (tuple):
                The predicted value.
            real (tuple):
                The actual value.
        """
        raise NotImplementedError('Please implement the loss measuring algorithm!')


class InverseCDFDistance(LossFunction):
    """Measure the distance between continuous key fields.

    This loss function first applies the fitted cdfs to every single entry (i.e. turning
    the numerical values into their respective percentiles) and then measures the Lp distance
    to the pth power, between the predicted value and the real value.

    Args:
        p (float):
            The p parameter in L_p metric. Must be positive.
    """

    def __init__(self, p=2):
        self.p = p
        self.cdfs = []

    def fit(self, data, cols):
        """Fits univariate distributions (automatically selected).

        Args:
            data (DataFrame):
                Data, where each column in `cols` is a continuous column.
            cols (list[str]):
                Column names.
        """
        for col in cols:
            col_data = np.array(data[col])
            dist_model = Univariate()
            dist_model.fit(col_data)
            self.cdfs.append(dist_model)

    def measure(self, pred, real):
        """Compute the distance (L_p norm) between the pred and real values.

        This uses the probability integral transform to map the pred/real values
        to a CDF value (between 0.0 and 1.0). Then, it computes the L_p norm
        between the CDF(pred) and CDF(real).

        Args:
            pred (tuple):
                Predicted value(s) corresponding to the columns specified in fit.
            real (tuple):
                Real value(s) corresponding to the columns specified in fit.

        Returns:
            float:
                The L_p norm of the CDF value.
        """
        assert len(pred) == len(real)

        dist = 0
        for idx in range(len(real)):
            percentiles = self.cdfs[idx].cdf(np.array([pred[idx], real[idx]]))
            dist += abs(percentiles[0] - percentiles[1]) ** self.p

        return dist

class Goal(Enum):
    """Goal Enumeration.

    This enumerates the ``goal`` for a metric; the value of a metric can be ignored,
    minimized, or maximized.
    """

    IGNORE = 'ignore'
    MAXIMIZE = 'maximize'
    MINIMIZE = 'minimize'

class CategoricalType(Enum):
    """Enumerates the type required for a categorical data.

    The value can be one-hot-encoded, or coded as class number.
    """

    CLASS_NUM = 'Class_num'
    ONE_HOT = 'One_hot'

class BaseMetric:
    """Base class for all the metrics in SDMetrics.

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

    name = None
    goal = None
    min_value = None
    max_value = None

    @classmethod
    def get_subclasses(cls, include_parents=False):
        """Recursively find subclasses of this metric.

        If ``include_parents`` is passed as ``True``, intermediate child classes
        that also have subclasses will be included. Otherwise, only classes
        without subclasses will be included to ensure that they are final
        implementations and are ready to be run on data.

        Args:
            include_parents (bool):
                Whether to include subclasses which are parents to
                other classes. Defaults to ``False``.
        """
        subclasses = {}
        for child in cls.__subclasses__():
            grandchildren = child.get_subclasses(include_parents)
            subclasses.update(grandchildren)
            if include_parents or not grandchildren:
                subclasses[child.__name__] = child

        return subclasses

    @staticmethod
    def compute(real_data, synthetic_data):
        """Compute this metric.

        Args:
            real_data:
                The values from the real dataset.
            synthetic_data:
                The values from the synthetic dataset.

        Returns:
            Union[float, tuple[float]]:
                Metric output or outputs.
        """
        raise NotImplementedError()

    @classmethod
    def normalize(cls, raw_score):
        """Compute the normalized value of the metric.

        Args:
            raw_score (float):
                The value of the metric from `compute`.

        Returns:
            float:
                The normalized value of the metric.
        """
        min_value = float(cls.min_value)
        max_value = float(cls.max_value)

        if max_value < raw_score or min_value > raw_score:
            raise ValueError('`raw_score` must be between `min_value` and `max_value`.')

        is_min_finite = min_value not in (float('-inf'), float('inf'))
        is_max_finite = max_value not in (float('-inf'), float('inf'))

        score = None
        if is_min_finite and is_max_finite:
            score = (raw_score - min_value) / (max_value - min_value)

        elif not is_min_finite and is_max_finite:
            score = np.exp(raw_score - max_value)

        elif is_min_finite and not is_max_finite:
            score = 1.0 - np.exp(min_value - raw_score)

        else:
            score = 1 / (1 + np.exp(-raw_score))

        if score is None or score < 0 or score > 1:
            raise AssertionError(
                f'This should be unreachable. The score {score} should bea value between 0 and 1.'
            )

        if cls.goal == Goal.MINIMIZE:
            return 1.0 - score

        return score


class SingleColumnMetric(BaseMetric):
    """Base class for metrics that apply to individual columns.

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

    name = None
    goal = None
    min_value = None
    max_value = None

    @staticmethod
    def compute(real_data, synthetic_data):
        """Compute this metric.

        Args:
            real_data (Union[numpy.ndarray, pandas.Series]):
                The values from the real dataset, passed as a 1d numpy
                array or as a pandas.Series.
            synthetic_data (Union[numpy.ndarray, pandas.Series]):
                The values from the synthetic dataset, passed as a 1d numpy
                array or as a pandas.Series.

        Returns:
            float
                Metric output.
        """
        raise NotImplementedError()

    @classmethod
    def compute_breakdown(cls, real_data, synthetic_data):
        """Compute this metric breakdown.

        Args:
            real_data (Union[numpy.ndarray, pandas.Series]):
                The values from the real dataset, passed as a 1d numpy
                array or as a pandas.Series.
            synthetic_data (Union[numpy.ndarray, pandas.Series]):
                The values from the synthetic dataset, passed as a 1d numpy
                array or as a pandas.Series.

        Returns:
            dict
                Mapping of the metric output. Must include the key 'score'.
        """
        return {'score': cls.compute(real_data, synthetic_data)}





def get_type_from_column_meta(column_metadata):
    """Get the type of a given column from the column metadata.

    Args:
        column_metadata (dict):
            The column metadata.

    Returns:
        string:
            The column type.
    """
    return column_metadata.get('sdtype', '')

def get_columns_from_metadata(metadata):
    """Get the column info from a metadata dict.

    Args:
        metadata (dict):
            The metadata dict.

    Returns:
        dict:
            The columns metadata.
    """
    return metadata.get('columns', {})

def nested_attrs_meta(nested):
    """Metaclass factory that defines a Metaclass with a dynamic attribute name."""

    class Metaclass(type):
        """Metaclass which pulls the attributes from a nested object using properties."""

        def __getattr__(cls, attr):
            """If cls does not have the attribute, try to get it from the nested object."""
            nested_obj = getattr(cls, nested)
            if hasattr(nested_obj, attr):
                return getattr(nested_obj, attr)

            raise AttributeError(f"type object '{cls.__name__}' has no attribute '{attr}'")

        @property
        def name(cls):
            return getattr(cls, nested).name

        @property
        def goal(cls):
            return getattr(cls, nested).goal

        @property
        def max_value(cls):
            return getattr(cls, nested).max_value

        @property
        def min_value(cls):
            return getattr(cls, nested).min_value

    return Metaclass

def get_alternate_keys(metadata):
    """Get the alternate keys from a metadata dict.

    Args:
        metadata (dict):
            The metadata dict.

    Returns:
        list:
            The list of alternate keys.
    """
    alternate_keys = []
    for alternate_key in metadata.get('alternate_keys', []):
        if isinstance(alternate_key, list):
            alternate_keys.extend(alternate_key)
        else:
            alternate_keys.append(alternate_key)

    return alternate_keys


def get_frequencies(real, synthetic):
    """Get percentual frequencies for each possible real categorical value.

    Given two iterators containing categorical data, this transforms it into
    observed/expected frequencies which can be used for statistical tests. It
    adds a regularization term to handle cases where the synthetic data contains
    values that don't exist in the real data.

    Args:
        real (list):
            A list of hashable objects.
        synthetic (list):
            A list of hashable objects.

    Yields:
        tuble[list, list]:
            The observed and expected frequencies (as a percent).
    """
    f_obs, f_exp = [], []
    real, synthetic = Counter(real), Counter(synthetic)
    for value in synthetic:
        if value not in real:
            real[value] += 1e-6  # Regularization to prevent NaN.

    for value in real:
        f_obs.append(synthetic[value] / sum(synthetic.values()))  # noqa: PD011
        f_exp.append(real[value] / sum(real.values()))  # noqa: PD011

    return f_obs, f_exp

def is_datetime(data):
    """Determine if the input is a datetime type or not.

    Args:
        data (pandas.DataFrame, int or datetime):
            Input to evaluate.

    Returns:
        bool:
            True if the input is a datetime type, False if not.
    """
    return (
        pd.api.types.is_datetime64_any_dtype(data)
        or isinstance(data, pd.Timestamp)
        or isinstance(data, datetime)
    )


def _validate_metadata_dict(metadata):
    """Validate the metadata type."""
    if not isinstance(metadata, dict):
        raise TypeError(
            f"Expected a dictionary but received a '{type(metadata).__name__}' instead."
            " For SDV metadata objects, please use the 'to_dict' function to convert it"
            ' to a dictionary.'
        )

def _validate_single_table_metadata(metadata):
    """Validate the metadata for a single table."""
    _validate_metadata_dict(metadata)
    if 'columns' not in metadata:
        raise ValueError(
            "Single-table metadata must include a 'columns' key that maps column names"
            ' to their corresponding information.'
        )

def _convert_datetime_column(column_name, column_data, column_metadata):
    if is_datetime(column_data):
        return column_data

    datetime_format = column_metadata.get('datetime_format')
    if datetime_format is None:
        raise ValueError(
            f"Datetime column '{column_name}' does not have a specified 'datetime_format'. "
            'Please add a the required datetime_format to the metadata or convert this column '
            "to 'pd.datetime' to bypass this requirement."
        )

    try:
        pd.to_datetime(column_data, format=datetime_format)
    except Exception as e:
        raise ValueError(f"Error converting column '{column_name}' to timestamp: {e}")

    return pd.to_datetime(column_data, format=datetime_format)


class IncomputableMetricError(Exception):
    """Raised when a metric cannot be computed."""


class ColumnPairsMetric(BaseMetric):
    """Base class for metrics that compare pairs of columns.

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

    name = None
    goal = None
    min_value = None
    max_value = None

    @staticmethod
    def compute(real_data, synthetic_data):
        """Compute this metric.

        Args:
            real_data (pandas.DataFrame):
                The values from the real dataset, passed as pandas.DataFrame
                with 2 columns.
            synthetic_data (pandas.DataFrame):
                The values from the synthetic dataset, passed as a
                pandas.DataFrame with 2 columns.

        Returns:
            float:
                Metric output.
        """
        raise NotImplementedError()

    @classmethod
    def compute_breakdown(cls, real_data, synthetic_data):
        """Compute the breakdown of this metric.

        Args:
            real_data (pandas.DataFrame):
                The values from the real dataset, passed as pandas.DataFrame
                with 2 columns.
            synthetic_data (pandas.DataFrame):
                The values from the synthetic dataset, passed as a
                pandas.DataFrame with 2 columns.

        Returns:
            dict
                A mapping of the metric output. Must contain the key 'score'.
        """
        return {'score': cls.compute(real_data, synthetic_data)}


class SingleTableMetric(BaseMetric):
    """Base class for metrics that apply to single tables.

    Input to these family of metrics are two ``pandas.DataFrame`` instances
    and a ``dict`` representations of the corresponding ``Table`` metadata.

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

    name = None
    goal = None
    min_value = None
    max_value = None

    _DTYPES_TO_TYPES = {
        'i': {'sdtype': 'numerical'},
        'f': {'sdtype': 'numerical'},
        'O': {'sdtype': 'categorical'},
        'b': {'sdtype': 'boolean'},
        'M': {'sdtype': 'datetime'},
    }

    @classmethod
    def _select_fields(cls, metadata, types):
        """Select fields from metadata that match the specified types.

        Args:
            metadata (dict):
                The table metadata.
            types (str or tuple):
                The desired data types.

        Returns:
            list:
                All fields that match the specified types.

        Raises:
            IncompatibleMetricError:
                If no matching fields are found, the metric is unable to be computed.
        """
        fields = []
        if isinstance(types, str):
            types = (types,)

        primary_key = metadata.get('primary_key', '')
        alternate_keys = get_alternate_keys(metadata)

        for field_name, field_meta in get_columns_from_metadata(metadata).items():
            if 'pii' in field_meta or field_name == primary_key or field_name in alternate_keys:
                continue

            field_type = get_type_from_column_meta(field_meta)
            field_subtype = field_meta.get('subtype')
            if any(t in types for t in (field_type, (field_type,), (field_type, field_subtype))):
                fields.append(field_name)

        if len(fields) == 0:
            raise IncomputableMetricError(f'Cannot find fields of types {types}')

        return fields

    @classmethod
    def _validate_inputs(cls, real_data, synthetic_data, metadata=None):
        """Validate the inputs and return the validated data and metadata.

        If a metadata is passed, the data is validated against it.

        If no metadata is passed, one is built based on the ``real_data`` values.

        Args:
            real_data (pandas.DataFrame):
                The real data.
            synthetic_data(pandas.DataFrame):
                The synthetic data.
            metadata (dict):
                The metadata, if any.

        Returns:
            (pandas.DataFrame, pandas.DataFrame, dict):
                The validated data and metadata.
        """
        real_data = real_data.copy()
        synthetic_data = synthetic_data.copy()
        if metadata is not None:
            metadata = copy.deepcopy(metadata)

        if set(real_data.columns) != set(synthetic_data.columns):
            raise ValueError('`real_data` and `synthetic_data` must have the same columns')

        if metadata is not None:
            _validate_single_table_metadata(metadata)
            fields = get_columns_from_metadata(metadata)
            for column in real_data.columns:
                if column not in fields:
                    raise ValueError(f'Column {column} not found in metadata')

            for field, field_meta in fields.items():
                field_type = get_type_from_column_meta(field_meta)
                if field not in real_data.columns:
                    raise ValueError(f'Field {field} not found in data')

                if field_type == 'datetime':
                    real_data[field] = _convert_datetime_column(field, real_data[field], field_meta)
                    synthetic_data[field] = _convert_datetime_column(
                        field, synthetic_data[field], field_meta
                    )

            return real_data, synthetic_data, metadata

        dtype_kinds = real_data.dtypes.apply(attrgetter('kind'))
        return (
            real_data,
            synthetic_data,
            {
                'columns': dtype_kinds.apply(cls._DTYPES_TO_TYPES.get).to_dict(),
            },
        )

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None):
        """Compute this metric.

        Real data and synthetic data must be passed as ``pandas.DataFrame`` instances
        and ``metadata`` as a ``Table`` metadata ``dict`` representation.

        If no ``metadata`` is given, one will be built from the values observed
        in the ``real_data``.

        Args:
            real_data (pandas.DataFrame):
                The values from the real dataset, passed as a pandas.DataFrame.
            synthetic_data (pandas.DataFrame):
                The values from the synthetic dataset, passed as a pandas.DataFrame.
            metadata (dict):
                Table metadata dict. If not passed, it is build based on the
                real_data fields and dtypes.

        Returns:
            Union[float, tuple[float]]:
                Metric output.
        """
        raise NotImplementedError()

    @classmethod
    def compute_breakdown(cls, real_data, synthetic_data, metadata=None):
        """Compute this metric breakdown.

        Args:
            real_data (pandas.DataFrame):
                The values from the real dataset, passed as a pandas.DataFrame.
            synthetic_data (pandas.DataFrame):
                The values from the synthetic dataset, passed as a pandas.DataFrame.
            metadata (dict):
                Table metadata dict. If not passed, it is build based on the
                real_data fields and dtypes.
            real_data (Union[numpy.ndarray, pandas.Series]):
                The values from the real dataset, passed as a 1d numpy
                array or as a pandas.Series.

        Returns:
            dict
                Mapping of the metric output. Must include the key 'score'.
        """
        return {'score': cls.compute(real_data, synthetic_data, metadata)}

class MultiSingleColumnMetric(
    SingleTableMetric, metaclass=nested_attrs_meta('single_column_metric')
):
    """SingleTableMetric subclass that applies a SingleColumnMetric on each column.

    This class can either be used by creating a subclass that inherits from it and
    sets the SingleColumn Metric as the ``single_column_metric`` attribute,
    or by creating an instance of this class passing the underlying SingleColumn
    metric as an argument.

    Attributes:
        name (str):
            Name to use when reports about this metric are printed.
        goal (sdmetrics.goal.Goal):
            The goal of this metric.
        min_value (Union[float, tuple[float]]):
            Minimum value or values that this metric can take.
        max_value (Union[float, tuple[float]]):
            Maximum value or values that this metric can take.
        single_column_metric (sdmetrics.single_column.base.SingleColumnMetric):
            SingleColumn metric to apply.
        field_types (dict):
            Field types to which the SingleColumn metric will be applied.
    """

    single_column_metric = None
    single_column_metric_kwargs = None
    field_types = None

    def __init__(self, single_column_metric=None, **single_column_metric_kwargs):
        self.single_column_metric = single_column_metric
        self.single_column_metric_kwargs = single_column_metric_kwargs
        self.compute = self._compute

    def _compute(self, real_data, synthetic_data, metadata=None, store_errors=False, **kwargs):
        """Compute this metric for all columns.

        This is done by computing the underlying SingleColumn metric to all the
        columns that are compatible with it.

        The output is a mapping of column name to the score of that column.

        Args:
            real_data (pandas.DataFrame):
                The values from the real dataset.
            synthetic_data (pandas.DataFrame):
                The values from the synthetic dataset.
            metadata (dict):
                Table metadata dict.
            store_errors (bool):
                Whether or not to store any metric computation errors in the results.
            **kwargs:
                Any additional keyword arguments will be passed down
                to the single column metric

        Returns:
            Dict[string -> Union[float, tuple[float]]]:
                A mapping of column name to metric output.
        """
        real_data, synthetic_data, metadata = self._validate_inputs(
            real_data, synthetic_data, metadata
        )

        fields = self._select_fields(metadata, self.field_types)
        invalid_cols = set(get_columns_from_metadata(metadata).keys()) - set(fields)

        scores = {col: {'score': np.nan} for col in invalid_cols}
        for column_name, real_column in real_data.items():
            if column_name in fields:
                real_column = real_column.to_numpy()
                synthetic_column = synthetic_data[column_name].to_numpy()

                try:
                    score = self.single_column_metric.compute_breakdown(
                        real_column,
                        synthetic_column,
                        **(self.single_column_metric_kwargs or {}),
                        **kwargs,
                    )
                    scores[column_name] = score
                except Exception as error:
                    if store_errors:
                        scores[column_name] = {'error': error}
                    else:
                        raise error

        return scores

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None, **kwargs):
        """Compute this metric.

        This is done by computing the underlying SingleColumn metric to all the
        columns that are compatible with it.

        The output is the average of the scores obtained.

        Args:
            real_data (pandas.DataFrame):
                The values from the real dataset.
            synthetic_data (pandas.DataFrame):
                The values from the synthetic dataset.
            metadata (dict):
                Table metadata dict.
            **kwargs:
                Any additional keyword arguments will be passed down
                to the single column metric

        Returns:
            Union[float, tuple[float]]:
                Metric output.
        """
        scores = cls._compute(cls, real_data, synthetic_data, metadata, **kwargs)
        return np.nanmean([breakdown['score'] for breakdown in scores.values()])

    @classmethod
    def compute_breakdown(cls, real_data, synthetic_data, metadata=None, **kwargs):
        """Compute this metric broken down by column.

        This is done by computing the underlying SingleColumn metric to all the
        columns that are compatible with it.

        The output is a mapping of column to the column's score.

        Args:
            real_data (pandas.DataFrame):
                The values from the real dataset.
            synthetic_data (pandas.DataFrame):
                The values from the synthetic dataset.
            metadata (dict):
                Table metadata dict.
            **kwargs:
                Any additional keyword arguments will be passed down
                to the single column metric

        Returns:
            Dict[string -> Union[float, tuple[float]]]:
                A mapping of column name to metric output.
        """
        return cls._compute(cls, real_data, synthetic_data, metadata, store_errors=True, **kwargs)

    @classmethod
    def normalize(cls, raw_score):
        """Return the `raw_score` as is, since it is already normalized.

        Args:
            raw_score (float):
                The value of the metric from `compute`.

        Returns:
            float:
                The normalized value of the metric
        """
        assert cls.min_value == 0.0
        return super().normalize(raw_score)


class MultiColumnPairsMetric(SingleTableMetric, metaclass=nested_attrs_meta('column_pairs_metric')):
    """SingleTableMetric subclass that applies a ColumnPairsMetric on each possible column pair.

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
            ColumnPairsMetric to apply.
        field_types (dict):
            Field types to which the SingleColumn metric will be applied.
    """

    column_pairs_metric = None
    column_pairs_metric_kwargs = None
    field_types = None

    def __init__(self, column_pairs_metric, **column_pairs_metric_kwargs):
        self.column_pairs_metric = column_pairs_metric
        self.column_pairs_metric_kwargs = column_pairs_metric_kwargs
        self.compute = self._compute

    def _compute(self, real_data, synthetic_data, metadata=None, **kwargs):
        """Compute this metric.

        This is done by grouping all the columns that are compatible with the
        underlying ColumnPairs metric in groups of 2 and then evaluating them
        using the ColumnPairs metric.

        The output is the average of the scores obtained.

        Args:
            real_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the synthetic dataset.
            metadata (dict):
                Table metadata dict.
            **kwargs:
                Any additional keyword arguments will be passed down
                to the column pairs metric

        Returns:
            Union[float, tuple[float]]:
                Metric output.
        """
        real_data, synthetic_data, metadata = self._validate_inputs(
            real_data, synthetic_data, metadata
        )

        fields = self._select_fields(metadata, self.field_types)

        values = []
        for columns in combinations(fields, r=2):
            real = real_data[list(columns)]
            synthetic = synthetic_data[list(columns)]
            values.append(self.column_pairs_metric.compute(real, synthetic))

        return np.nanmean(values)

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None, **kwargs):
        """Compute this metric.

        Args:
            real_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the synthetic dataset.
            metadata (dict):
                Table metadata dict.
            **kwargs:
                Any additional keyword arguments will be passed down
                to the column pairs metric

        Returns:
            Union[float, tuple[float]]:
                Metric output.
        """
        return cls._compute(cls, real_data, synthetic_data, metadata, **kwargs)

    @classmethod
    def compute_breakdown(cls, real_data, synthetic_data, metadata=None, **kwargs):
        """Compute the breakdown of this metric.

        Args:
            real_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the synthetic dataset.
            metadata (dict):
                Table metadata dict.
            **kwargs:
                Any additional keyword arguments will be passed down
                to the column pairs metric

        Returns:
            dict:
                Metric output.
        """
        real_data, synthetic_data, metadata = cls._validate_inputs(
            real_data, synthetic_data, metadata
        )

        fields = cls._select_fields(metadata, cls.field_types)

        breakdown = {}
        for columns in combinations(fields, r=2):
            sorted_columns = tuple(sorted(columns))
            real = real_data[list(sorted_columns)]
            synthetic = synthetic_data[list(sorted_columns)]
            breakdown[sorted_columns] = cls.column_pairs_metric.compute_breakdown(
                real, synthetic, **kwargs
            )

        return breakdown

    @classmethod
    def normalize(cls, raw_score):
        """Return the `raw_score` as is, since it is already normalized.

        Args:
            raw_score (float):
                The value of the metric from `compute`.

        Returns:
            float:
                The normalized value of the metric
        """
        assert cls.min_value == 0.0
        return super().normalize(raw_score)


class CategoricalPrivacyMetric(SingleTableMetric):
    """Base class for Categorical Privacy metrics on single tables.

    These metrics fit an adversial attacker model on the synthetic data and
    then evaluate its accuracy (or probability of making the correct attack)
    on the real data.

    Attributes:
        name (str):
            Name to use when reports about this metric are printed.
        goal (sdmetrics.goal.Goal):
            The goal of this metric.
        min_value (Union[float, tuple[float]]):
            Minimum value or values that this metric can take.
        max_value (Union[float, tuple[float]]):
            Maximum value or values that this metric can take.
        model:
            Model class to use for the prediction.
        model_kwargs:
            Keyword arguments to use to create the model instance.
        accuracy_base (bool):
            True if the privacy score should be based on the accuracy of the attacker,
            False if it should be based on the probability of making the correct attack.
    """

    name = None
    goal = Goal.MAXIMIZE
    min_value = 0
    max_value = 1
    MODEL = None
    MODEL_KWARGS = {}
    ACCURACY_BASE = None

    @classmethod
    def _fit(cls, synthetic_data, key_fields, sensitive_fields, model_kwargs):
        if model_kwargs is None:
            model_kwargs = cls.MODEL_KWARGS.copy() if cls.MODEL_KWARGS else {}

        model = cls.MODEL(**model_kwargs)
        model.fit(synthetic_data, key_fields, sensitive_fields)
        return model

    @classmethod
    def _validate_inputs(cls, real_data, synthetic_data, metadata, key_fields, sensitive_fields):
        real_data, synthetic_data, metadata = super()._validate_inputs(
            real_data, synthetic_data, metadata
        )
        if 'key_fields' in metadata:
            key_fields = metadata['key_fields']
        elif key_fields is None:
            raise TypeError('`key_fields` must be passed either directly or inside `metadata`')

        if 'sensitive_fields' in metadata:
            sensitive_fields = metadata['sensitive_fields']
        elif sensitive_fields is None:
            raise TypeError(
                '`sensitive_fields` must be passed either directly or inside `metadata`'
            )

        if len(key_fields) == 0 or len(sensitive_fields) == 0:
            raise ValueError('`key_fields` or `sensitive_fields` is empty')

        return key_fields, sensitive_fields, metadata

    @classmethod
    def compute(
        cls,
        real_data,
        synthetic_data,
        metadata=None,
        key_fields=None,
        sensitive_fields=None,
        model_kwargs=None,
    ):
        """Compute this metric.

        This fits an adversial attacker model on the synthetic data and
        then evaluates it making predictions on the real data.

        A ``key_fields`` column(s) name must be given, either directly or as a first level
        entry in the ``metadata`` dict, which will be used as the key column(s) for the
        attack.

        A ``sensitive_fields`` column(s) name must be given, either directly or as a first level
        entry in the ``metadata`` dict, which will be used as the sensitive_fields column(s)
        for the attack.

        Args:
            real_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the synthetic dataset.
            metadata (dict):
                Table metadata dict. If not passed, it is build based on the
                real_data fields and dtypes.
            key_fields (list(str)):
                Name of the column(s) to use as the key attributes.
            sensitive_fields (list(str)):
                Name of the column(s) to use as the sensitive attributes.
            model_kwargs (dict):
                Key word arguments of the attacker model. cls.MODEL_KWARGS will be used
                if none is provided.

        Returns:
            union[float, tuple[float]]:
                Scores obtained by the attackers when evaluated on the real data.
        """
        key_fields, sensitive_fields, metadata = cls._validate_inputs(
            real_data, synthetic_data, metadata, key_fields, sensitive_fields
        )

        for col in key_fields + sensitive_fields:
            data_type = get_columns_from_metadata(metadata)[col]
            if (
                data_type != cls._DTYPES_TO_TYPES['i']
                and data_type != cls._DTYPES_TO_TYPES['O']
                and data_type != cls._DTYPES_TO_TYPES['b']
            ):  # check data type
                raise TypeError(f'Column {col} has invalid data type {data_type}')

        model = cls._fit(synthetic_data, key_fields, sensitive_fields, model_kwargs)

        if cls.ACCURACY_BASE:  # calculate privacy score based on prediction accuracy
            count = len(real_data)
            match = 0
            for idx in range(count):
                key_data = tuple(real_data[key_fields].iloc[idx])
                sensitive_data = tuple(real_data[sensitive_fields].iloc[idx])
                pred_sensitive = model.predict(key_data)
                if pred_sensitive == sensitive_data:
                    match += 1

            return 1.0 - match / count

        else:  # calculate privacy score based on posterior prob of the correct sensitive data
            count = 0
            score = 0
            for idx in range(len(real_data)):
                key_data = tuple(real_data[key_fields].iloc[idx])
                sensitive_data = tuple(real_data[sensitive_fields].iloc[idx])
                row_score = model.score(key_data, sensitive_data)
                if row_score is not None:
                    count += 1
                    score += row_score

            if count == 0:
                return np.nan

            return 1.0 - score / count


class NumericalPrivacyMetric(SingleTableMetric):
    """Base class for Numerical Privacy metrics on single tables.

    These metrics fit an adversial attacker model on the synthetic data and
    then evaluate its accuracy (or probability of making the correct attack)
    on the real data.

    Attributes:
        name (str):
            Name to use when reports about this metric are printed.
        goal (sdmetrics.goal.Goal):
            The goal of this metric.
        min_value (Union[float, tuple[float]]):
            Minimum value or values that this metric can take.
        max_value (Union[float, tuple[float]]):
            Maximum value or values that this metric can take.
        model (Class):
            Model class to use for the prediction.
        model_kwargs (dict):
            Keyword arguments to use to create the model instance.
        loss_function (Class):
            Loss function to use when evaluating the accuracy of the privacy attack.
        loss_function_kwargs (dict):
            Keyword arguments to use to create the loss function instance.
    """

    name = None
    goal = Goal.MAXIMIZE
    min_value = 0
    max_value = np.inf
    MODEL = None
    MODEL_KWARGS = {}
    LOSS_FUNCTION = InverseCDFDistance
    LOSS_FUNCTION_KWARGS = {'p': 2}

    @classmethod
    def _fit(cls, synthetic_data, key_fields, sensitive_fields, model_kwargs):
        if model_kwargs is None:
            model_kwargs = cls.MODEL_KWARGS.copy() if cls.MODEL_KWARGS else {}

        model = cls.MODEL(**model_kwargs)
        model.fit(synthetic_data, key_fields, sensitive_fields)

        return model

    @classmethod
    def _validate_inputs(cls, real_data, synthetic_data, metadata, key_fields, sensitive_fields):
        real_data, synthetic_data, metadata = super()._validate_inputs(
            real_data, synthetic_data, metadata
        )
        if 'key_fields' in metadata:
            key_fields = metadata['key_fields']
        elif key_fields is None:
            raise TypeError('`key_fields` must be passed either directly or inside `metadata`')

        if 'sensitive_fields' in metadata:
            sensitive_fields = metadata['sensitive_fields']
        elif sensitive_fields is None:
            raise TypeError(
                '`sensitive_fields` must be passed either directly or inside `metadata`'
            )

        if len(key_fields) == 0 or len(sensitive_fields) == 0:
            raise ValueError('`key_fields` or `sensitive_fields` is empty')

        return key_fields, sensitive_fields, metadata

    @classmethod
    def compute(
        cls,
        real_data,
        synthetic_data,
        metadata=None,
        key_fields=None,
        sensitive_fields=None,
        model_kwargs=None,
        loss_function=None,
        loss_function_kwargs=None,
    ):
        """Compute this metric.

        This fits an adversial attacker model on the synthetic data and
        then evaluates it making predictions on the real data.

        A ``key_fields`` column(s) name must be given, either directly or as a first level
        entry in the ``metadata`` dict, which will be used as the key column(s) for the
        attack.

        A ``sensitive_fields`` column(s) name must be given, either directly or as a first level
        entry in the ``metadata`` dict, which will be used as the sensitive column(s) for the
        attack.

        Args:
            real_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the synthetic dataset.
            metadata (dict):
                Table metadata dict. If not passed, it is build based on the
                real_data fields and dtypes.
            key_fields (list(str)):
                Name of the column(s) to use as the key attributes.
            sensitive_fields (list(str)):
                Name of the column(s) to use as the sensitive attributes.
            model_kwargs (dict):
                Key word arguments of the attacker model. cls.MODEL_KWARGS will be used
                if none is provided.
            loss_function (Class):
                The loss function to use. cls.LOSS_FUNCTION will be used if none is provided.
            loss_function_kwargs (dict):
                Key word arguments of the loss function. cls.LOSS_FUNCTION_KWARGS will be used
                if none is provided.

        Returns:
            union[float, tuple[float]]:
                Scores obtained by the attackers when evaluated on the real data.
        """
        key_fields, sensitive_fields, metadata = cls._validate_inputs(
            real_data, synthetic_data, metadata, key_fields, sensitive_fields
        )

        for col in key_fields + sensitive_fields:
            data_type = get_columns_from_metadata(metadata)[col]

            # check data type
            if data_type != cls._DTYPES_TO_TYPES['i'] and data_type != cls._DTYPES_TO_TYPES['f']:
                raise TypeError(f'Column {col} has invalid data type {data_type}')

        model = cls._fit(synthetic_data, key_fields, sensitive_fields, model_kwargs)

        if loss_function_kwargs is None:
            loss_function_kwargs = cls.LOSS_FUNCTION_KWARGS

        if loss_function is None:
            loss_function = cls.LOSS_FUNCTION(**loss_function_kwargs)
        else:
            loss_function = loss_function(**loss_function_kwargs)

        loss_function.fit(real_data, sensitive_fields)

        count = len(real_data)
        score = 0
        for idx in range(count):
            key_data = tuple(real_data[key_fields].iloc[idx])
            sensitive_data = tuple(real_data[sensitive_fields].iloc[idx])
            pred_sensitive = model.predict(key_data)
            score += loss_function.measure(pred_sensitive, sensitive_data)

        return score / count


class PrivacyAttackerModel:
    """Train and evaluate a privacy model.

    Train a model to predict sensitive attributes from key attributes
    using the synthetic data. Then, evaluate the privacy of the model by
    trying to predict the sensitive attributes of the real data.
    """

    def fit(self, synthetic_data, key_fields, sensitive_fields):
        """Fit the attacker on the synthetic data.

        Args:
            synthetic_data(pandas.DataFrame):
                The synthetic data table used for adverserial learning.
            key_fields(list[str]):
                The names of the key columns.
            sensitive_fields(list[str]):
                The names of the sensitive columns.
        """
        raise NotImplementedError('Please implement fit method of attackers')

    def predict(self, key_data):
        """Make a prediction of the sensitive data given keys.

        Args:
            key_data(tuple):
                The key data.

        Returns:
            tuple:
                The predicted sensitive data.
        """
        raise NotImplementedError('Please implement predict method of attackers')

    def score(self, key_data, sensitive_data):
        """Score based on the belief of the attacker, in the form P(sensitive_data|key|data).

        Args:
            key_data(tuple):
                The key data.
            sensitive_data(tuple):
                The sensitive data.
        """
        raise NotImplementedError(
            'Posterior probability based scoring not supportedfor this attacker!'
        )
    

MODELABLE_SDTYPES = ('numerical', 'datetime', 'categorical', 'boolean')


def _validate_metadata_dict(metadata):
    """Validate the metadata type."""
    if not isinstance(metadata, dict):
        raise TypeError(
            f"Expected a dictionary but received a '{type(metadata).__name__}' instead."
            " For SDV metadata objects, please use the 'to_dict' function to convert it"
            ' to a dictionary.'
        )


def _validate_single_table_metadata(metadata):
    """Validate the metadata for a single table."""
    _validate_metadata_dict(metadata)
    if 'columns' not in metadata:
        raise ValueError(
            "Single-table metadata must include a 'columns' key that maps column names"
            ' to their corresponding information.'
        )


def _validate_multi_table_metadata(metadata):
    """Validate the metadata for multiple tables."""
    _validate_metadata_dict(metadata)
    if 'tables' not in metadata:
        raise ValueError(
            "Multi-table metadata must include a 'tables' key that maps table names"
            ' to their respective metadata.'
        )
    for table_name, table_metadata in metadata['tables'].items():
        try:
            _validate_single_table_metadata(table_metadata)
        except ValueError as e:
            raise ValueError(f"Error in table '{table_name}': {str(e)}")


def _validate_metadata(metadata):
    """Validate the metadata."""
    _validate_metadata_dict(metadata)
    if ('columns' not in metadata) and ('tables' not in metadata):
        raise ValueError(
            "Metadata must include either a 'columns' key for single-table metadata"
            " or a 'tables' key for multi-table metadata."
        )

    if 'tables' in metadata:
        _validate_multi_table_metadata(metadata)


def handle_single_and_multi_table(single_table_func):
    """Decorator to handle both single and multi table functions."""

    def wrapper(data, metadata):
        if isinstance(data, pd.DataFrame):
            return single_table_func(data, metadata)

        result = {}
        for table_name in data:
            result[table_name] = single_table_func(data[table_name], metadata['tables'][table_name])

        return result

    return wrapper


def _convert_datetime_column(column_name, column_data, column_metadata):
    if is_datetime(column_data):
        return column_data

    datetime_format = column_metadata.get('datetime_format')
    if datetime_format is None:
        raise ValueError(
            f"Datetime column '{column_name}' does not have a specified 'datetime_format'. "
            'Please add a the required datetime_format to the metadata or convert this column '
            "to 'pd.datetime' to bypass this requirement."
        )

    try:
        pd.to_datetime(column_data, format=datetime_format)
    except Exception as e:
        raise ValueError(f"Error converting column '{column_name}' to timestamp: {e}")

    return pd.to_datetime(column_data, format=datetime_format)


@handle_single_and_multi_table
def _convert_datetime_columns(data, metadata):
    """Convert datetime columns to datetime type."""
    for column in metadata['columns']:
        if metadata['columns'][column]['sdtype'] == 'datetime':
            data[column] = _convert_datetime_column(
                column, data[column], metadata['columns'][column]
            )

    return data


@handle_single_and_multi_table
def _remove_missing_columns_metadata(data, metadata):
    """Remove columns that are not present in the metadata."""
    columns_in_metadata = set(metadata['columns'].keys())
    columns_in_data = set(data.columns)
    columns_to_remove = columns_in_data - columns_in_metadata
    extra_metadata_columns = columns_in_metadata - columns_in_data
    if columns_to_remove:
        columns_to_print = "', '".join(sorted(columns_to_remove))
        warnings.warn(
            f"The columns ('{columns_to_print}') are not present in the metadata. "
            'They will not be included for further evaluation.',
            UserWarning,
        )
    elif extra_metadata_columns:
        columns_to_print = "', '".join(sorted(extra_metadata_columns))
        warnings.warn(
            f"The columns ('{columns_to_print}') are in the metadata but they are not "
            'present in the data.',
            UserWarning,
        )

    data = data.drop(columns=columns_to_remove)
    column_intersection = [column for column in data.columns if column in metadata['columns']]

    return data[column_intersection]


@handle_single_and_multi_table
def _remove_non_modelable_columns(data, metadata):
    """Remove columns that are not modelable.

    All modelable columns are numerical, datetime, categorical, or boolean sdtypes.
    """
    columns_modelable = []
    for column in metadata['columns']:
        column_sdtype = metadata['columns'][column]['sdtype']
        if column_sdtype in MODELABLE_SDTYPES and column in data.columns:
            columns_modelable.append(column)

    return data[columns_modelable]



def _process_data_with_metadata(data, metadata, keep_modelable_columns_only=False):
    """Process the data according to the metadata."""
    _validate_metadata_dict(metadata)
    data = _convert_datetime_columns(data, metadata)
    data = _remove_missing_columns_metadata(data, metadata)
    if keep_modelable_columns_only:
        data = _remove_non_modelable_columns(data, metadata)

    return data

CHUNK_SIZE = 1000


def _process_dcr_chunk(dataset_chunk, reference_chunk, cols_to_keep, metadata, ranges):
    full_dataset = dataset_chunk.merge(reference_chunk, how='cross', suffixes=('_data', '_ref'))

    for col_name in cols_to_keep:
        sdtype = metadata['columns'][col_name]['sdtype']
        ref_column = full_dataset[col_name + '_ref']
        data_column = full_dataset[col_name + '_data']
        diff_col_name = col_name + '_diff'
        if sdtype in ['numerical', 'datetime']:
            diff = (ref_column - data_column).abs()
            if pd.api.types.is_timedelta64_dtype(diff):
                diff = diff.dt.total_seconds()

            full_dataset[col_name + '_diff'] = np.where(
                ranges[col_name] == 0,
                (diff > 0).astype(int),
                np.minimum(diff / ranges[col_name], 1.0),
            )

            xor_condition = (ref_column.isna() & ~data_column.isna()) | (
                ~ref_column.isna() & data_column.isna()
            )

            full_dataset.loc[xor_condition, diff_col_name] = 1

            both_nan_condition = ref_column.isna() & data_column.isna()

            full_dataset.loc[both_nan_condition, diff_col_name] = 0

        elif sdtype in ['categorical', 'boolean']:
            equals_cat = (ref_column == data_column) | (ref_column.isna() & data_column.isna())
            full_dataset[diff_col_name] = (~equals_cat).astype(int)

        full_dataset.drop(columns=[col_name + '_ref', col_name + '_data'], inplace=True)

    full_dataset['diff'] = full_dataset.iloc[:, 2:].sum(axis=1) / len(cols_to_keep)
    chunk_result = (
        full_dataset[['index_data', 'diff']].groupby('index_data').min().reset_index(drop=True)
    )
    return chunk_result['diff']


def calculate_dcr(dataset, reference_dataset, metadata, chunk_size=1000):
    """Calculate the Distance to Closest Record for all rows in the synthetic data.

    Arguments:
        dataset (pandas.Dataframe):
            The dataset for which we want to compute the DCR values
        reference_dataset (pandas.Dataframe):
            The reference dataset that is used for the distance computations
        metadata (dict):
            The metadata dict.

    Returns:
        pandas.Series:
            Returns a Series that shows the DCR value for every row of dataset
    """
    dataset = _process_data_with_metadata(dataset.copy(), metadata, True)
    reference = _process_data_with_metadata(reference_dataset.copy(), metadata, True)

    common_cols = set(dataset.columns) & set(reference.columns)
    cols_to_keep = []
    ranges = {}

    for col_name, col_metadata in get_columns_from_metadata(metadata).items():
        sdtype = col_metadata['sdtype']

        if (
            sdtype in ['numerical', 'categorical', 'boolean', 'datetime']
            and col_name in common_cols
        ):
            cols_to_keep.append(col_name)

            if sdtype in ['numerical', 'datetime']:
                col_range = reference[col_name].max() - reference[col_name].min()
                if isinstance(col_range, pd.Timedelta):
                    col_range = col_range.total_seconds()

                ranges[col_name] = col_range

    if not cols_to_keep:
        raise ValueError('There are no overlapping statistical columns to measure.')

    dataset = dataset[cols_to_keep]
    dataset['index'] = range(len(dataset))

    reference = reference[cols_to_keep]
    reference['index'] = range(len(reference))
    results = []

    for dataset_chunk_start in range(0, len(dataset), chunk_size):
        dataset_chunk = dataset.iloc[dataset_chunk_start : dataset_chunk_start + chunk_size]
        minimum_chunk_distance = None
        for reference_chunk_start in range(0, len(reference), chunk_size):
            reference_chunk = reference.iloc[
                reference_chunk_start : reference_chunk_start + chunk_size
            ]
            chunk_result = _process_dcr_chunk(
                dataset_chunk=dataset_chunk,
                reference_chunk=reference_chunk,
                cols_to_keep=cols_to_keep,
                metadata=metadata,
                ranges=ranges,
            )
            if minimum_chunk_distance is None:
                minimum_chunk_distance = chunk_result
            else:
                minimum_chunk_distance = pd.Series.min(
                    pd.concat([minimum_chunk_distance, chunk_result], axis=1), axis=1
                )

        results.append(minimum_chunk_distance)

    result = pd.concat(results, ignore_index=True)
    result.name = None

    return result