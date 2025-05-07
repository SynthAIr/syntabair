# This file includes code adapted from the SDMetrics project:
# https://github.com/sdv-dev/SDMetrics
# Licensed under the MIT License.
# Modifications have been made to suit this project's requirements.
import copy
from operator import attrgetter
import numpy as np
import pandas as pd
from enum import Enum
from datetime import datetime
from collections import Counter
from itertools import combinations
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder



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


class Goal(Enum):
    """Goal Enumeration.

    This enumerates the ``goal`` for a metric; the value of a metric can be ignored,
    minimized, or maximized.
    """

    IGNORE = 'ignore'
    MAXIMIZE = 'maximize'
    MINIMIZE = 'minimize'


class HyperTransformer:
    """HyperTransformer class.

    The ``HyperTransformer`` class contains a set of transforms to transform one or
    more columns based on each column's data type.
    """

    column_transforms = {}
    column_kind = {}

    def fit(self, data):
        """Fit the HyperTransformer to the given data.

        Args:
            data (pandas.DataFrame):
                The data to transform.
        """
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        for field in data:
            kind = data[field].dropna().infer_objects().dtype.kind
            self.column_kind[field] = kind

            if kind == 'i' or kind == 'f':
                # Numerical column.
                self.column_transforms[field] = {'mean': data[field].mean()}
            elif kind == 'b':
                # Boolean column.
                numeric = pd.to_numeric(data[field], errors='coerce').astype(float)
                self.column_transforms[field] = {'mode': numeric.mode().iloc[0]}
            elif kind == 'O':
                # Categorical column.
                col_data = pd.DataFrame({'field': data[field]})
                enc = OneHotEncoder()
                enc.fit(col_data)
                self.column_transforms[field] = {'one_hot_encoder': enc}
            elif kind == 'M':
                # Datetime column.
                nulls = data[field].isna()
                integers = pd.to_numeric(data[field], errors='coerce').to_numpy().astype(np.float64)
                integers[nulls] = np.nan
                self.column_transforms[field] = {'mean': pd.Series(integers).mean()}

    def transform(self, data):
        """Transform the given data based on the data type of each column.

        Args:
            data (pandas.DataFrame):
                The data to transform.

        Returns:
            pandas.DataFrame:
                The transformed data.
        """
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        for field in data:
            transform_info = self.column_transforms[field]

            kind = self.column_kind[field]
            if kind == 'i' or kind == 'f':
                # Numerical column.
                data[field] = data[field].fillna(transform_info['mean'])
            elif kind == 'b':
                # Boolean column.
                data[field] = pd.to_numeric(data[field], errors='coerce').astype(float)
                data[field] = data[field].fillna(transform_info['mode'])
            elif kind == 'O':
                # Categorical column.
                col_data = pd.DataFrame({'field': data[field]})
                out = transform_info['one_hot_encoder'].transform(col_data).toarray()
                transformed = pd.DataFrame(
                    out, columns=[f'{field}_value{i}' for i in range(np.shape(out)[1])]
                )
                data = data.drop(columns=[field])
                data = pd.concat([data, transformed.set_index(data.index)], axis=1)
            elif kind == 'M':
                # Datetime column.
                nulls = data[field].isna()
                integers = pd.to_numeric(data[field], errors='coerce').to_numpy().astype(np.float64)
                integers[nulls] = np.nan
                data[field] = pd.Series(integers)
                data[field] = data[field].fillna(transform_info['mean'])

        return data

    def fit_transform(self, data):
        """Fit and transform the given data based on the data type of each column.

        Args:
            data (pandas.DataFrame):
                The data to transform.

        Returns:
            pandas.DataFrame:
                The transformed data.
        """
        self.fit(data)
        return self.transform(data)


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

"""Base class for metrics that compare pairs of columns."""


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

class DetectionMetric(SingleTableMetric):
    """Base class for Machine Learning Detection based metrics on single tables.

    These metrics build a Machine Learning Classifier that learns to tell the synthetic
    data apart from the real data, which later on is evaluated using Cross Validation.

    The output of the metric is one minus the average ROC AUC score obtained.

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

    name = 'SingleTable Detection'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @staticmethod
    def _fit_predict(X_train, y_train, X_test):
        """Fit a classifier and then use it to predict."""
        raise NotImplementedError()

    @staticmethod
    def _drop_non_compute_columns(real_data, synthetic_data, metadata):
        """Drop all columns that cannot be statistically modeled."""
        transformed_real_data = real_data
        transformed_synthetic_data = synthetic_data

        if metadata is not None:
            drop_columns = []
            drop_columns.extend(get_alternate_keys(metadata))
            for column in metadata.get('columns', []):
                if 'primary_key' in metadata and (
                    column == metadata['primary_key'] or column in metadata['primary_key']
                ):
                    drop_columns.append(column)

                column_info = metadata['columns'].get(column, {})
                sdtype = column_info.get('sdtype')
                pii = column_info.get('pii')
                if sdtype not in ['numerical', 'datetime', 'categorical'] or pii:
                    drop_columns.append(column)

            if drop_columns:
                transformed_real_data = real_data.drop(drop_columns, axis=1)
                transformed_synthetic_data = synthetic_data.drop(drop_columns, axis=1)
        return transformed_real_data, transformed_synthetic_data

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None):
        """Compute this metric.

        This builds a Machine Learning Classifier that learns to tell the synthetic
        data apart from the real data, which later on is evaluated using Cross Validation.

        The output of the metric is one minus the average ROC AUC score obtained.

        Args:
            real_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the synthetic dataset.
            metadata (dict):
                Table metadata dict. If not passed, it is build based on the
                real_data fields and dtypes.

        Returns:
            float:
                One minus the ROC AUC Cross Validation Score obtained by the classifier.
        """
        real_data, synthetic_data, metadata = cls._validate_inputs(
            real_data, synthetic_data, metadata
        )

        transformed_real_data, transformed_synthetic_data = cls._drop_non_compute_columns(
            real_data, synthetic_data, metadata
        )

        ht = HyperTransformer()
        transformed_real_data = ht.fit_transform(transformed_real_data).to_numpy()
        transformed_synthetic_data = ht.transform(transformed_synthetic_data).to_numpy()
        X = np.concatenate([transformed_real_data, transformed_synthetic_data])
        y = np.hstack([
            np.ones(len(transformed_real_data)),
            np.zeros(len(transformed_synthetic_data)),
        ])
        if np.isin(X, [np.inf, -np.inf]).any():
            X[np.isin(X, [np.inf, -np.inf])] = np.nan

        try:
            scores = []
            kf = StratifiedKFold(n_splits=3, shuffle=True)
            for train_index, test_index in kf.split(X, y):
                y_pred = cls._fit_predict(X[train_index], y[train_index], X[test_index])
                roc_auc = roc_auc_score(y[test_index], y_pred)

                scores.append(max(0.5, roc_auc) * 2 - 1)

            return 1 - np.mean(scores)
        except ValueError as err:
            raise IncomputableMetricError(f'DetectionMetric: Unable to be fit with error {err}')

    @classmethod
    def normalize(cls, raw_score):
        """Return the `raw_score` as is, since it is already normalized.

        Args:
            raw_score (float):
                The value of the metric from `compute`.

        Returns:
            float:
                Simply returns `raw_score`.
        """
        return super().normalize(raw_score)    

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
