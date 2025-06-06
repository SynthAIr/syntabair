# This file includes code adapted from the REaLTabFormer project:
# https://github.com/worldbank/REaLTabFormer
# Licensed under the MIT License
# Modifications have been made to suit this project's requirements.
import math
import os
import random
from typing import Any, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm.auto import tqdm


class SyntheticDataBench:
    """This class handles all the assessments
    needed for testing the synthetic data.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        categorical: bool,
        target_pos_val: Any = None,
        test_size: float = 0.2,
        test_df: Optional[pd.DataFrame] = None,
        random_state: int = 1029,
    ) -> None:
        assert (
            test_size < 1
        ), "The test_size must be a fraction of the data, and should be less than 1."
        self.random_state = random_state

        # Target column in the data for the ML efficiency
        # measure.
        self.target_col = target_col
        self.categorical = categorical
        self.target_pos_val = target_pos_val

        if test_df is not None:
            self.test_df = test_df
            self.train_df = data
            self.test_size = None
        else:
            self.test_df = data.sample(
                frac=test_size, replace=False, random_state=self.random_state
            )
            self.train_df = data.loc[data.index.difference(self.test_df.index)]
            self.test_size = test_size

        self.train_df = pd.concat(
            [self.train_df.drop(target_col, axis=1), self.train_df[target_col]], axis=1
        )
        self.test_df = pd.concat(
            [self.test_df.drop(target_col, axis=1), self.test_df[target_col]], axis=1
        )

        self.n_test: int = len(self.test_df)
        self.n_train: int = len(self.train_df)
        self.synth_train_df: pd.DataFrame = None
        self.synth_test_df: pd.DataFrame = None

    def register_synthetic_data(self, synthetic: pd.DataFrame):
        """
        Registers synthetic data for the assessment.

        The synthetic data is split into training and test sets according
        to the values of n_train and n_test. The split is done by sampling
        the data without replacement.

        Args:
        synthetic: A DataFrame containing synthetic data.
        The DataFrame must have at least as many rows as n_train + n_test.

        Returns:
        None
        """

        assert synthetic.shape[0] >= (self.n_test + self.n_train)
        self.synth_train_df = synthetic.sample(
            n=self.n_train, replace=False, random_state=self.random_state
        )
        self.synth_test_df = synthetic.loc[
            synthetic.index.difference(self.synth_train_df.index)
        ].sample(n=self.n_test, replace=False, random_state=self.random_state)

    @staticmethod
    def compute_distance_to_closest_records(
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        n_test: int,
        distance: manhattan_distances = manhattan_distances,
    ) -> pd.Series:
        """
        original: The dataframe of the training data used to train the generative model.
        synthetic: The dataframe generated by the generative model or any data we want to compare
         with the original data.
        n_test: The number of observations we want to compare with the original data
         from the synthetic data. Ideally, this should be the same size as the test data.
        """
        assert n_test <= len(synthetic)
        synthetic = synthetic.iloc[:n_test]

        distances: np.ndarray = distance(original, synthetic)
        return pd.Series(distances.min(axis=0), index=synthetic.index)

    @staticmethod
    def measure_ml_efficiency(
        model: sklearn.base.BaseEstimator,
        train: pd.DataFrame,
        synthetic: pd.DataFrame,
        test: pd.DataFrame,
        target_col: str,
        random_state: int = 1029,
    ) -> pd.DataFrame:
        """
        This function trains the provided model on the original and synthetic training data,
        and then uses the trained models to make predictions on the test data. It returns a
        dataframe containing the actual values and predictions from both training sets. This
        dataframe can be used to compare the performance of the model trained on the original
        data with the model trained on the synthetic data.

        Parameters:
        model (sklearn.base.BaseEstimator): The model to be trained and used for prediction.
        train (pd.DataFrame): The original training data.
        synthetic (pd.DataFrame): The synthetic training data generated by a generative model.
         Must have the same size as the `train`.
        test (pd.DataFrame): The test data to be used for prediction.
        target_col (str): The name of the target column in the train and test data.

        Returns:
        pd.DataFrame: A dataframe containing the actual values and predictions from both
         training sets.
        """

        random.seed(random_state)
        np.random.seed(random_state)

        assert train.shape[0] == synthetic.shape[0]

        # Train the model on the original training data
        model.fit(train.drop(target_col, axis=1), train[target_col])

        # Make predictions on the test data using the original training data
        try:
            original_predictions = model.predict_proba(test.drop(target_col, axis=1))
        except AttributeError:
            original_predictions = model.predict(test.drop(target_col, axis=1))

        # Train the model on the synthetic training data
        model.fit(synthetic.drop(target_col, axis=1), synthetic[target_col])

        # Make predictions on the test data using the synthetic training data
        try:
            synthetic_predictions = model.predict_proba(test.drop(target_col, axis=1))
        except AttributeError:
            synthetic_predictions = model.predict(test.drop(target_col, axis=1))

        # Return a dataframe with the actual values and predictions from both training sets
        return pd.DataFrame(
            {
                "actual": test[target_col],
                "original_predictions": original_predictions,
                "synthetic_predictions": synthetic_predictions,
            }
        )

    @staticmethod
    def preprocess_data(
        data: pd.DataFrame,
        other: Union[pd.DataFrame, List[pd.DataFrame]] = None,
        fillna: bool = True,
    ) -> dict:
        """Preprocesses a DataFrame containing mixed data types and returns a feature matrix.

        The function first extracts the categorical and numerical columns from the DataFrame,
        and then applies a processing pipeline that one-hot encodes the categorical features
        and standardizes the numerical features.

        Args:
            data (pandas.DataFrame): A DataFrame containing mixed data types.

        Returns:
            dict:
             - preprocessor: The trained feature processor pipeline.
             - column_names: The new column names for the processed data.
             - data: A feature matrix containing only numerical values for the input data.
             - other (optional): A feature matrix containing only numerical values
              for the input other.
        """
        # Define a processing pipeline for the data
        index = data.index
        numeric_features = data.select_dtypes(include="number").columns
        categorical_features = data.select_dtypes(include="object").columns

        transformers = []
        column_names = []
        numeric_cols = []
        categorical_cols = []

        if not numeric_features.empty:
            numeric_transformer = StandardScaler()
            transformers.append(("num", numeric_transformer, numeric_features))
            column_names.extend(numeric_features)
            numeric_cols.extend(numeric_features)

        if not categorical_features.empty:
            categorical_transformer = OneHotEncoder(handle_unknown="ignore")
            transformers.append(("cat", categorical_transformer, categorical_features))

        preprocessor = ColumnTransformer(transformers=transformers)
        preprocessor.fit(data)

        if not categorical_features.empty:
            for transf in preprocessor.transformers_:
                if isinstance(transf[1], OneHotEncoder):
                    column_names.extend(transf[1].get_feature_names_out())
                    categorical_cols.extend(transf[1].get_feature_names_out())

        data = preprocessor.transform(data)

        payload = dict(
            preprocessor=preprocessor,
            numeric_cols=numeric_cols,
            column_names=column_names,
            categorical_cols=categorical_cols,
            data=pd.DataFrame(
                # Sometimes the transform method returns a sparse matrix.
                # So we convert it to an array if the data is not a numpy
                # array.
                data if isinstance(data, np.ndarray) else data.toarray(),
                columns=column_names,
                index=index,
            ),
        )
        if fillna:
            payload["data"] = payload["data"].fillna(payload["data"].mean())

        if other is not None:
            transformed_other = []
            is_df = False

            if isinstance(other, pd.DataFrame):
                is_df = True
                other = [other]

            for _other in other:
                index = _other.index
                _other = preprocessor.transform(_other)

                _other = pd.DataFrame(
                    _other if isinstance(_other, np.ndarray) else _other.toarray(),
                    columns=column_names,
                    index=index,
                )
                if fillna:
                    _other = _other.fillna(_other.mean())
                transformed_other.append(_other)

            payload["other"] = transformed_other[0] if is_df else transformed_other

        return payload

    @staticmethod
    def compute_discriminator_predictions(
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        test: pd.DataFrame,
        model: sklearn.base.BaseEstimator,
        random_state: int = 1029,
    ) -> dict:
        """
        Builds a discriminator model that attempts to distinguish between original and synthetic data.

        The function first preprocesses the data by extracting the categorical and numerical columns,
        then applies a processing pipeline that one-hot encodes the categorical features
        and standardizes the numerical features.
        Next, it adds labels to the original and synthetic data to indicate which is which,
        then combines the data into one DataFrame and splits it into training and test sets.
        Finally, it trains a classifier model on the training data and returns the model.

        Args:
            original (pandas.DataFrame): A DataFrame containing original data.
            synthetic (pandas.DataFrame): A DataFrame containing synthetic data.
            model (Type[LogisticRegression]): A type of scikit-learn model to use.
                Defaults to LogisticRegression.
            test_size (float): The proportion of data to include in the test set.
                Defaults to 0.2.
            random_state (int): The random seed to use for splitting the data.
                Defaults to 1029.

        Returns:
            dict:
             - y_test: Labels for the test/synthetic test data.
             - y_preds: Predictions for the label.
        """
        assert synthetic.shape[0] >= (len(original) + len(test))

        train_synthetic = synthetic.sample(
            n=len(original), replace=False, random_state=random_state
        )
        test_synthetic = synthetic.loc[
            synthetic.index.difference(train_synthetic.index)
        ].sample(n=len(test), replace=False, random_state=random_state)

        # Preprocess the original and synthetic data
        processed = SyntheticDataBench.preprocess_data(
            data=original, other=[train_synthetic, test, test_synthetic]
        )
        original = processed["data"]
        train_synthetic = processed["other"][0]
        test = processed["other"][1]
        test_synthetic = processed["other"][2]

        label_col_name = "discriminator_label"

        assert label_col_name not in processed["column_names"]

        # Add labels to the original and synthetic data
        original[label_col_name] = 1
        train_synthetic[label_col_name] = 0

        # Combine the original and synthetic data
        combined_data = pd.concat([original, train_synthetic])

        feat_train = combined_data.drop(label_col_name, axis=1)
        y_train = combined_data[label_col_name]
        feat_train = feat_train.fillna(feat_train.mean())

        # Train a logistic regression model on the training data
        model.fit(feat_train, y_train)

        oob_score = None
        try:
            oob_score = model.oob_score_
            print("Discriminator OOB Score:", oob_score)
        except AttributeError:
            pass

        # Transform the test data and the synthetic test data
        test[label_col_name] = 1
        test_synthetic[label_col_name] = 0

        combined_test = pd.concat([test, test_synthetic])
        feat_test = combined_test.drop(label_col_name, axis=1)
        y_test = combined_test[label_col_name]
        feat_test = feat_test.fillna(feat_test.mean())

        # Estimate the discriminator on the test data
        y_preds = model.predict(feat_test)

        return dict(y_test=y_test, y_preds=y_preds, oob_score=oob_score)

    def get_dcr(
        self, is_test: bool = False, distance: manhattan_distances = manhattan_distances
    ) -> pd.Series:
        """Get the DCR values for this experiment."""
        train = self.train_df
        other = self.test_df if is_test else self.synth_train_df

        if (train.dtypes == "object").any():
            proc = self.preprocess_data(data=train, other=other)
            train = proc["data"]
            other = proc["other"]

        return self.compute_distance_to_closest_records(
            train,
            other,
            self.n_test,
            distance=distance,
        )

    def get_ml_efficiency(
        self, model: sklearn.base.BaseEstimator, synthetic: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Get the ML efficiency for this experiment."""
        train = self.train_df.copy()
        test = self.test_df.copy()

        if synthetic is None:
            synthetic = self.synth_train_df

        train_target = train[self.target_col]
        test_target = test[self.target_col]
        synthetic_target = synthetic[self.target_col]

        if self.categorical:
            train_target = 1 * (self.target_pos_val == train_target)
            test_target = 1 * (self.target_pos_val == test_target)
            synthetic_target = 1 * (self.target_pos_val == synthetic_target)

        train = train.drop(self.target_col, axis=1)
        test = test.drop(self.target_col, axis=1)
        synthetic = synthetic.drop(self.target_col, axis=1)

        proc = self.preprocess_data(data=train, other=[test, synthetic])
        train = proc["data"]
        test = proc["other"][0]
        synthetic = proc["other"][1]

        train = pd.concat([train, train_target], axis=1)
        test = pd.concat([test, test_target], axis=1)
        synthetic = pd.concat([synthetic, synthetic_target], axis=1)

        return self.measure_ml_efficiency(
            model=model,
            train=train,
            synthetic=synthetic,
            test=test,
            target_col=self.target_col,
            random_state=self.random_state,
        )

    def get_discriminator_performance(self, model: sklearn.base.BaseEstimator):
        """Compute the discriminator performance for this experiment."""
        return self.compute_discriminator_predictions(
            original=self.train_df,
            synthetic=pd.concat([self.synth_train_df, self.synth_test_df]),
            test=self.test_df,
            model=model,
            random_state=self.random_state,
        )

    @staticmethod
    def compute_data_copying_predictions(
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        test: pd.DataFrame,
        model: sklearn.base.BaseEstimator,
        random_state: int = 1029,
    ) -> dict:
        """
        Builds a discriminator model that attempts to distinguish between original and synthetic data.

        The function first preprocesses the data by extracting the categorical and numerical columns,
        then applies a processing pipeline that one-hot encodes the categorical features
        and standardizes the numerical features.
        Next, it adds labels to the original and synthetic data to indicate which is which,
        then combines the data into one DataFrame and splits it into training and test sets.
        Finally, it trains a classifier model on the training data and returns the model.

        Args:
            original (pandas.DataFrame): A DataFrame containing original data.
            synthetic (pandas.DataFrame): A DataFrame containing synthetic data.
            model (Type[LogisticRegression]): A type of scikit-learn model to use.
                Defaults to LogisticRegression.
            test_size (float): The proportion of data to include in the test set.
                Defaults to 0.2.
            random_state (int): The random seed to use for splitting the data.
                Defaults to 1029.

        Returns:
            dict:
             - y_test: Labels for the test/synthetic test data.
             - y_preds: Predictions for the label.
        """
        # assert synthetic.shape[0] >= (len(original) + len(test))

        # Preprocess the original and test data
        processed = SyntheticDataBench.preprocess_data(original, test)
        preprocessor = processed["preprocessor"]
        column_names = processed["column_names"]
        original = processed["data"]
        test = processed["other"]

        label_col_name = "discriminator_label"

        assert label_col_name not in processed["column_names"]

        # Add labels to the original and test data
        original[label_col_name] = 1
        test[label_col_name] = 0

        # Combine the original and test data
        combined_data = pd.concat([original, test])

        X_train = combined_data.drop(label_col_name, axis=1)
        y_train = combined_data[label_col_name]
        X_train = X_train.fillna(X_train.mean())

        # Train a logistic regression model on the training data
        model.fit(X_train, y_train)

        try:
            print("Discriminator OOB Score:", model.oob_score_)
        except AttributeError:
            pass

        # Transform the test data and the synthetic test data
        synthetic = preprocessor.transform(synthetic)
        synthetic = pd.DataFrame(
            synthetic if isinstance(synthetic, np.ndarray) else synthetic.toarray(),
            columns=column_names,
        )
        X_test = synthetic
        X_test = X_test.fillna(X_test.mean())

        # Estimate the discriminator on the test data
        y_preds = model.predict(X_test)

        return dict(y_preds=y_preds)

    @staticmethod
    def compute_sensitivity_metric(
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
        test: pd.DataFrame,
        qt_max: float = 0.05,
        qt_interval: int = 1000,
        distance: manhattan_distances = manhattan_distances,
        tsvd: TruncatedSVD = None,
        max_col_nums: int = 50,
        use_ks: bool = False,
        verbose: bool = False,
    ) -> float:
        object_dtypes = original.select_dtypes(exclude="number")

        if not object_dtypes.empty or original.shape[1] >= max_col_nums:
            if verbose:
                print("Transforming data with non-numeric values...")
            processed = SyntheticDataBench.preprocess_data(
                original, other=[test, synthetic]
            )

            original = processed["data"]
            test = processed["other"][0]
            synthetic = processed["other"][1]

        if tsvd is None and original.shape[1] >= max_col_nums:
            # We use a truncated SVD with components at least 5 or
            # equivalent to the square root of the original number
            # of variables.
            tsvd = TruncatedSVD(
                n_components=max(5, math.ceil(original.shape[1] ** 0.5))
            )

        if tsvd is not None:
            if verbose:
                print("Applying TruncatedSVD")
            # We reduce the dimensionality of the transformed data
            original = tsvd.fit_transform(original)
            test = tsvd.transform(test)
            synthetic = tsvd.transform(synthetic)

        # We compute the relative distances of each sub data with respect
        # to the original data.
        test_distances: np.ndarray = distance(original, test)
        synth_distances: np.ndarray = distance(original, synthetic)

        # We take the distance of the closest point from each sub data
        # to observations in the original data.
        test_min = test_distances.min(axis=1)
        synth_min = synth_distances.min(axis=1)

        tr_test_min = test_distances.min(axis=0)
        tr_synth_min = synth_distances.min(axis=0)

        # We don't include observations that have duplicates in both
        # the synthetic and the test data.
        fltr = (synth_min == 0) & (test_min == 0)
        test_min = test_min[~fltr]
        synth_min = synth_min[~fltr]

        test_min = np.concatenate([test_min, tr_test_min])
        synth_min = np.concatenate([synth_min, tr_synth_min])

        statistic = None

        if use_ks:
            test_min = test_min[test_min <= np.quantile(test_min, qt_max)]
            synth_min = synth_min[synth_min <= np.quantile(synth_min, qt_max)]

            ks_stat = stats.ks_2samp(test_min, synth_min)

            statistic = ks_stat.statistic

        elif qt_interval <= 1:
            statistic = (synth_min <= np.quantile(test_min, qt_max)).mean()
        else:
            # We define the quantile set to assess the systematic
            # bias in the distance of the synthetic data, if any.
            quantiles = np.linspace(0, qt_max, qt_interval)

            # # The vectorized form is equivalent to the expanded form below:
            # # We do not use the absolute value of the difference so that the
            # # asymptotic value should be closer to 0. Anything that is significantly
            # # different from 0 is anomalous.
            # np.mean([((synth_min <= np.quantile(test_min, qt)).mean() - qt) for qt in np.linspace(0, qt_max, qt_interval)])

            # For each quantile of distances from the test data, we take the proportion
            # of the synthetic data with distance values lower than the value in the given quantile.
            # We expect that the more the model becomes overfitted, the closer to zero the distances
            # coming from the synthetic data is. However, if the data comes from the same distribution,
            # we expect to see this statistic to be closer to zero.
            # We use `<=` so that we can still capture the statistic correctly even when
            # the quantile value is zero.
            statistic = np.mean(
                (
                    synth_min.reshape(1, -1)
                    <= np.quantile(test_min, quantiles).reshape(-1, 1)
                ).mean(axis=1)
                - quantiles
            )

        return statistic

    @staticmethod
    def compute_sensitivity_threshold(
        train_data: pd.DataFrame,
        num_bootstrap: int = 100,
        test_size: int = None,
        frac: float = None,
        qt_max: float = 0.05,
        qt_interval: int = 1000,
        distance: manhattan_distances = manhattan_distances,
        tsvd: TruncatedSVD = None,
        return_values: bool = False,
        quantile: float = 0.95,
        max_col_nums: int = 50,
        use_ks: bool = False,
        full_sensitivity: bool = True,
        sensitivity_orig_frac_multiple: int = 3,
    ) -> Union[float, List]:
        """This method implements a bootstrapped estimation of the
        sensitivity values derived from the training data.

        We compute the sensitivity value for `num_bootstrap` rounds of random split
        of the training data.

        Args:
            quantile: Returns the sensitivity value at the given quantile from
             the bootstrap set. Note that we use quantile > 0.5 because we want to
             detect whether the synthetic data tends to be closer to the training data
             than expected. The statistic computes synth_min < test_min, so if the
             synthetic data systematically copies observation from the training data,
             we expect that the statictic tends to become larger >> 0.
            return_values: Instead of returning a single value based on the `quantile`
             argument, return the full set of boostrap values.
            sensitivity_orig_frac_multiple: The size of the training data relative to the chosen `frac` that will be
             used in computing the sensitivity. The larger this value is, the more robust the sensitivity threshold
             will be. However, `(sensitivity_orig_frac_multiple + 2)` multiplied by `frac` must be less than 1.
        """
        if test_size is not None:
            assert (
                test_size > 1
            ), "The test_size argument corresponds to the number of test samples"
            frac = test_size / len(train_data)

        if frac is None:
            raise ValueError("Either the test_size or frac must be provided")

        assert (
            2 * frac
        ) < 1, "This exceeds the test size and no training data will remain."

        values = []

        if not full_sensitivity:
            # We use the full fraction
            frac = 2 * frac

            # We will be sampling data from the dataset that is
            # `(sensitivity_orig_frac_multiple + 2) * frac` without
            # replacement. We should make sure that we don't exceed this
            # threshold.
            assert (sensitivity_orig_frac_multiple + 2) * frac <= 1

        def bootstrap_inner_loop():
            original: pd.DataFrame = None
            test: pd.DataFrame = None

            if full_sensitivity:
                original, test = train_test_split(train_data, test_size=2 * frac)
                synthetic = test.iloc[: len(test) // 2]
                test = test.loc[test.index.difference(synthetic.index)]
            else:
                source: pd.DataFrame = None
                n_size = int(len(train_data) * frac)
                n_train_size = sensitivity_orig_frac_multiple * n_size
                test_size = n_train_size + (2 * n_size)

                _, source = train_test_split(train_data, test_size=test_size)
                original = source.iloc[:n_train_size]
                synthetic = source.iloc[n_train_size : n_train_size + n_size]
                test = source.iloc[n_train_size + n_size :]

                assert synthetic.shape[0] == test.shape[0]

            return SyntheticDataBench.compute_sensitivity_metric(
                original=original,
                synthetic=synthetic,
                test=test,
                qt_max=qt_max,
                qt_interval=qt_interval,
                distance=distance,
                tsvd=tsvd,
                max_col_nums=max_col_nums,
                use_ks=use_ks,
            )

        n_jobs = 1
        cpu_count = os.cpu_count()

        if cpu_count and cpu_count >= 4:
            # n_jobs = min(max(2, cpu_count // 4), 16)
            n_jobs = min(max(2, cpu_count // 8), 16)

        if n_jobs == 1:
            for _ in tqdm(range(num_bootstrap), desc="Bootstrap round"):
                values.append(bootstrap_inner_loop())
        else:
            print("Using parallel computation!!!")
            with joblib.Parallel(n_jobs=n_jobs) as parallel:
                values = parallel(
                    joblib.delayed(bootstrap_inner_loop)()
                    for _ in tqdm(range(num_bootstrap), desc="Bootstrap round")
                )

        print("Sensitivity threshold summary:")
        print(pd.Series(values).describe())

        return values if return_values else np.quantile(values, quantile)


class SyntheticDataExperiment:
    """
    For each data and model:
    1. Split train/test data -> save data
    2. Train model with train data -> save model
    3. Generate N x train+test synthetic data -> save samples
    4. Perform analysis on the generated data.
    """

    def __init__(
        self,
        data_id: str,
        model_type: str,
        categorical: bool,
        target_col: str,
        target_pos_val: Any = None,
    ) -> None:
        pass
