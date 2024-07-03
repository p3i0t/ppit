from __future__ import annotations
from typing import Protocol, TYPE_CHECKING, Self

if TYPE_CHECKING:
    import polars as pl

# P = ParamSpec("P")
# T = TypeVar("T")

__all__ = ["DataFrameNormalizer"]


class DataFrameNormalizer(Protocol):
    "scikit-learn transformer-like protocol for normalizing DataFrame."

    def reset(self) -> None:
        ...

    def fit(self, data: pl.DataFrame) -> Self:
        ...

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        ...

    def fit_transform(self, data: pl.DataFrame) -> pl.DataFrame:
        ...
