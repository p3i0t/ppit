from typing import Protocol, Optional
from collections import namedtuple
from dataclasses import dataclass
import datetime

import polars as pl
import numpy as np

from config import Config


class DataArguments:
    pass


CryptoDataset = namedtuple("CryptoDataset", ["symbol", "time", "x", "y"])


@dataclass
class CryptoDatasets:
    train: CryptoDataset
    eval: CryptoDataset
    test: Optional[CryptoDataset] = None


class DataSource(Protocol):
    name = "_datasource_"

    def __init__(self, data_args: DataArguments) -> None:
        super().__init__()

    def __call__(self) -> CryptoDatasets:
        ...

    def _to_dataset(self) -> CryptoDataset:
        ...


class CryptoDataArguments(DataArguments):
    def __init__(
        self,
        x_path: Optional[str] = None,
        y_path: Optional[str] = None,
        x_columns: Optional[list[str]] = None,
        y_columns: Optional[list[str]] = None,
        signal_freq="1h",
        milestone: Optional[datetime.datetime] = None,
        n_train_days: int = 1000,
        n_eval_days: int = 30,
        n_test_days: int = 0,
        n_lag_days: int = 5,
        normalization: str = "zscore",
    ) -> None:
        super().__init__()

        cfg = Config()
        self.x_path = x_path or cfg.downsampled_10m_dir
        self.y_path = y_path or cfg.ret_1h_dir
        self.x_columns = x_columns or cfg.features_agg
        if y_columns is None:
            raise ValueError("y_columns is required")
        self.y_columns = y_columns
        self.signal_freq = signal_freq
        if milestone is None:
            raise ValueError("milestone is required")
        self.milestone = milestone
        self.n_train_days = n_train_days
        self.n_eval_days = n_eval_days
        self.n_test_days = n_test_days
        self.n_lag_days = n_lag_days
        self.normalization = normalization

        self.train_range = (
            milestone
            - datetime.timedelta(days=n_lag_days + n_eval_days + n_train_days),
            milestone - datetime.timedelta(days=n_lag_days + n_eval_days),
        )
        self.eval_range = (
            milestone - datetime.timedelta(days=n_lag_days + n_eval_days),
            milestone - datetime.timedelta(days=n_lag_days),
        )
        self.test_range = (
            milestone - datetime.timedelta(days=n_lag_days),
            milestone
            - datetime.timedelta(days=n_lag_days)
            + datetime.timedelta(days=n_test_days),
        )


class CryptoDataSource(DataSource):
    def __init__(self, data_args: CryptoDataArguments) -> None:
        super().__init__(data_args)
        self.data_args = data_args
        self.normalizer = get_normalizer(self.data_args.normalization)

        train_begin, train_end = self.data_args.train_range
        self.df_train_lazy = pl.scan_parquet(self.data_args.path).filter(
            pl.col("time").is_between(pl.lit(train_begin), pl.lit(train_end))
        )

        eval_begin, eval_end = self.data_args.eval_range
        self.df_eval_lazy = pl.scan_parquet(self.data_args.path).filter(
            pl.col("time").is_between(pl.lit(eval_begin), pl.lit(eval_end))
        )

        if self.data_args.n_test_days > 0:
            test_begin, test_end = self.data_args.test_range
            self.df_test_lazy = pl.scan_parquet(self.data_args.path).filter(
                pl.col("time").is_between(pl.lit(test_begin), pl.lit(test_end))
            )

    def _to_dataset(self, df: pl.DataFrame) -> CryptoDataset:
        """Convert a DataFrame to a CryptoDataset."""
        df_pack = (
            df.group_by_dynamic(
                "time",
                every=self.data_args.signal_freq,
                label="right",
                group_by="symbol",
            ).agg(
                *self.data_args.x_columns,
                *[pl.col(yc).last() for yc in self.data_args.y_columns],
                pl.col("time").first().alias("begin"),
                pl.col("time").last().alias("end"),
                pl.col("time").count().alias("count"),
            )
            # .filter(pl.col("count") == 6)
            # .with_columns(*[pl.col(c).list.to_array(6) for c in cols])
        )

        x = np.stack(
            [
                np.array(df_pack.get_column(c).to_list())
                for c in self.data_args.x_columns
            ],
            axis=-1,
        )

        return CryptoDataset(
            symbol=df_pack["symbol"].to_numpy(),
            time=df_pack["time"].to_numpy(),
            x=x,
            y=df_pack["close"],
        )

    def __call__(self) -> CryptoDatasets:
        df_train = self.normalizer.fit_transform(self.df_train_lazy.collect())
        df_eval = self.normalizer.transform(self.df_eval_lazy.collect())

        if self.data_args.n_test_days > 0:
            df_test = self.normalizer(self.df_test_lazy.collect())

        return CryptoDatasets(
            train=self._to_dataset(df_train),
            eval=self._to_dataset(df_eval),
            test=self._to_dataset(df_test) if self.data_args.n_test_days > 0 else None,
        )
