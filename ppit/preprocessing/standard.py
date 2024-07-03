from typing import Optional
import polars as pl
from .transformer_base import DataFrameNormalizer


class StandardScaler(DataFrameNormalizer):
    """Standardize features by removing the mean and scaling to unit variance.

    Args:
        kept_columns (Optional[list[str]], optional): columns should be existing and kept during normalization. Defaults to None.
    """

    def __init__(self, kept_columns: Optional[list[str]] = None):
        self.kept_columns = kept_columns or []

    def reset(self):
        if hasattr(self, "mean"):
            delattr(self, "mean")
            delattr(self, "std")
            delattr(self, "n_samples_seen")
            delattr(self, "columns")

    def __repr__(self):
        return (
            f"StandardScaler("
            f"n_samples_seen={getattr(self, 'n_samples_seen', None)}, "
            f"columns={getattr(self, 'columns', None)})"
        )

    # def _get_valid_columns()
    def partial_fit(self, x: pl.DataFrame):
        if not set(self.kept_columns).issubset(set(x.columns)):
            raise ValueError(
                f"required columns: {self.kept_columns} missing in input dataframe."
            )
        x_normalize = x.select(pl.col(pl.NUMERIC_DTYPES).exclude(self.kept_columns))

        setattr(self, "n_samples_seen", len(x))
        setattr(self, "mean", x_normalize.mean())
        setattr(self, "std", x_normalize.std())
        setattr(self, "columns", x_normalize.columns)
        return self

    def fit(self, x: pl.DataFrame):
        self.reset()
        return self.partial_fit(x=x)

    def transform(self, x: pl.DataFrame) -> pl.DataFrame:
        if not set(self.kept_columns).issubset(set(x.columns)):
            raise ValueError(
                f"required columns: {self.kept_columns} missing in input dataframe."
            )
        columns = x.select(pl.col(pl.NUMERIC_DTYPES).exclude(self.kept_columns)).columns

        columns_diff = set(columns) - set(getattr(self, "columns"))
        if len(columns_diff) > 0:
            raise ValueError(
                f"{columns_diff} are extra columns not fitted by this normalizer."
            )

        x = x.with_columns(
            pl.col(_c)
            .sub(getattr(self, "mean")[_c].item())
            .truediv(getattr(self, "std")[_c].item())
            for _c in columns
        )
        return x.select(self.kept_columns + columns)

    def fit_transform(self, x: pl.DataFrame) -> pl.DataFrame:
        return self.fit(x).transform(x)
