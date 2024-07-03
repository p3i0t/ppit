from typing import Optional
import polars as pl
from .transformer_base import DataFrameNormalizer


class GroupStandardScaler(DataFrameNormalizer):
    "todo: docstring"

    def __init__(
        self, group_columns: list[str] | str, kept_columns: Optional[list[str]] = None
    ):
        self.group_columns = (
            group_columns if isinstance(group_columns, list) else [group_columns]
        )
        self.kept_columns = kept_columns or []

    def reset(self):
        if hasattr(self, "columns"):
            delattr(self, "columns")

    def __repr__(self):
        return f"GroupStandardScaler(" f"columns={getattr(self, 'columns', None)})"

    def partial_fit(self, x: pl.DataFrame):
        if not set(self.kept_columns).issubset(set(x.columns)):
            raise ValueError(
                f"required columns: {self.kept_columns} missing in input dataframe."
            )
        columns = x.select(pl.col(pl.NUMERIC_DTYPES).exclude(self.kept_columns)).columns

        setattr(self, "n_samples_seen", len(x))
        setattr(self, "columns", columns)
        return self

    def fit(self, x: pl.DataFrame):
        self.reset()
        return self.partial_fit(x=x)

    def transform(self, x: pl.DataFrame) -> pl.DataFrame:
        """Normalize all numerical columns of input dataframe, except for the required columns.

        Args:
            x (pl.DataFrame): input dataframe.

        Raises:
            ValueError: required columns missing in input dataframe.
            ValueError: input dataframe contains unseen columns for this normalizer.

        Returns:
            pl.DataFrame: output dataframe.
        """
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
            pl.col(c).sub(pl.col(c).mean()).truediv(pl.col(c).std()).over("date")
            for c in columns
        )
        return x.select(self.kept_columns + columns)

    def fit_transform(self, x: pl.DataFrame) -> pl.DataFrame:
        return self.fit(x).transform(x)
