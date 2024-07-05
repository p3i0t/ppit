import itertools
from enum import Enum
from typing import Optional

import polars as pl


class Aggregator(str, Enum):
    MEAN = "mean"
    STD = "std"
    SKEW = "skew"
    KURT = "kurt"
    ZSCORE = "zscore"


def generate_agg_expr(col: str, agg: Aggregator) -> pl.Expr:
    if agg == "mean":
        return pl.col(col).mean().alias(f"{col}_mean")
    elif agg == "std":
        return pl.col(col).std().alias(f"{col}_std")
    elif agg == "skew":
        return pl.col(col).skew().alias(f"{col}_skew")
    elif agg == "kurt":
        return pl.col(col).kurtosis().alias(f"{col}_kurt")
    elif agg == "zscore":
        return pl.col(col).sub(pl.col(col).mean()).truediv(pl.col(col).std()).last().alias(f"{col}_zscore")
    else:
        raise ValueError(f"Unknown aggregation: {agg}")


def downsample_1m(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    freq_in_minute: int = 10,
    bars: Optional[list[str]] = None,
    agg_list: Optional[list[Aggregator]] = None,
) -> pl.DataFrame:
    required_cols = ["time", "symbol", "date"]
    if not set(required_cols).issubset(df.columns):
        raise ValueError(f"Required columns: {required_cols}")
    
    if bars is None:
        bars = [_col for _col in df.columns if _col not in required_cols]

    if agg_list is None:
        agg_list = [Aggregator.MEAN, Aggregator.STD, Aggregator.SKEW, Aggregator.KURT, Aggregator.ZSCORE]

    exprs = [
        generate_agg_expr(bar, agg) for bar, agg in itertools.product(bars, agg_list)
    ]
    downsampled_expr_names = [expr.meta.output_name() for expr in exprs]

    exprs.append(pl.col("time").count().alias("count"))  # for debug

    if isinstance(df, pl.DataFrame):
        df = df.lazy()

    df = df.select(required_cols + bars).sort(by=required_cols)
    freq = f"{freq_in_minute}m"
    _df = (
        df.group_by_dynamic(
            index_column="time",
            every=freq,
            period=freq,
            # offset="1m",
            label="right",
            group_by=["symbol", "date"],
            closed="left",
            include_boundaries=True,
        )
        .agg(exprs)
        .filter(pl.col("count") == freq_in_minute)
        .with_columns(pl.col(pl.NUMERIC_DTYPES).cast(pl.Float32))
        .collect()
    )

    _df: pl.DataFrame = _df.with_columns(
        pl.col("_upper_boundary").dt.strftime("%H%M").alias("slot"),
    )
    _df = _df.pivot(index=["symbol", "date"], columns="slot", values=downsampled_expr_names)
    _df = _df.rename({c: c.replace("_slot", "") for c in _df.columns})
    _df = _df.with_columns(pl.col("symbol").cast(pl.Categorical))
    return _df
