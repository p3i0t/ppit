import os
import datetime

import polars as pl
import typer

from .downsample import downsample_1m_to_10m
from .config import Config

app = typer.Typer()

RAY_IP = '127.0.0.1'
RAY_PORT = 8778

RAY_ADDRESS = f"ray://{RAY_IP}:{RAY_PORT}"

@app.command()
def down(n_parallel: int = 10):
    "downsample 1m bars to 10m bars."
    cfg = Config()

    import ray
    import glob

    # ray.init(address=RAY_ADDRESS, num_cpus=n_parallel, ignore_reinit_error=True)

    @ray.remote(num_cpus=1)
    def _run_single(src_file: str):
        df = pl.read_parquet(src_file)
        df_down = downsample_1m_to_10m(
            df, cfg.time_col, cfg.by_cols, bars=cfg.features_raw, agg_list=None
        )
        if not os.path.exists(cfg.downsampled_10m_dir):
            os.makedirs(cfg.downsampled_10m_dir, exist_ok=True)
        df_down.write_parquet(
            os.path.join(cfg.downsampled_10m_dir, os.path.basename(src_file))
        )

    src_files = glob.glob(os.path.join(cfg.bar_1m_dir, "*.parquet"))
    typer.echo(f"Total files to process: {len(src_files)}")
    MAX_QUEUE_SIZE = n_parallel
    task_ids = []
    queue_results = []
    for src_file in src_files:
        task_ids.append(_run_single.remote(src_file))
        if len(task_ids) >= MAX_QUEUE_SIZE:
            done_ids, task_ids = ray.wait(task_ids, num_returns=1)
            queue_results.extend(ray.get(done_ids))
    queue_results.extend(ray.get(task_ids))
    typer.echo("Done")


@app.command()
def download_ret(n_parallel: int = 10):
    "download return for every 1h."
    cfg = Config()

    import ray
    import glob
    from pathlib import Path

    @ray.remote(num_cpus=1)
    def run_single(begin):
        import datareader as dr
        dr.CONFIG.database_url["crypto"] = "clickhouse://allread:read_only@10.25.3.20:9100/binance"
        df = dr.read(
            dr.m.CryptoUMMinuteReturn(
                "1h",
                duration="1m",
                price="close",
                future=True,
                with_n_offsets=[
                    1,
                    2,
                    3,
                    4,
                    6,
                    12,
                    18,
                    24,
                    24 * 2,
                    24 * 3,
                ],
                apply_funding_rate=True,
            ),
            begin=begin,
            end=begin,
            df_lib="polars",
            categorical_symbol=True,
        )
        if not os.path.exists(cfg.ret_1h_dir):
            os.makedirs(cfg.ret_1h_dir, exist_ok=True)
        df.write_parquet(
            os.path.join(cfg.ret_1h_dir, f"{begin:%Y-%m-%d}.parquet")
        )

    dates = pl.date_range(
        start=datetime.date(2020, 1, 1), 
        end=datetime.datetime.today().date(), 
        interval="1d", 
        eager=True
    ).to_list()
    
    existing_ret_dates = [Path(_p).stem for _p in glob.glob(os.path.join(cfg.ret_1h_dir, "*.parquet"))]
    typer.echo(f"Total existing dates: {len(existing_ret_dates)}") 
    ret_left_dates = [_d for _d in dates if _d.strftime("%Y-%m-%d") not in existing_ret_dates]  
    
    typer.echo(f"Total dates to process: {len(ret_left_dates)}")
    import ray
    # ray.init(address=RAY_ADDRESS, num_cpus=n_parallel, num_gpus=0, ignore_reinit_error=True)
    MAX_QUEUE_SIZE = n_parallel
    task_ids = []
    queue_results = []
    for _d in ret_left_dates:
        task_ids.append(run_single.remote(_d))
        if len(task_ids) >= MAX_QUEUE_SIZE:
            done_ids, task_ids = ray.wait(task_ids, num_returns=1)
            queue_results.extend(ray.get(done_ids))
    queue_results.extend(ray.get(task_ids))
    typer.echo("Done")


@app.command()
def gen_dataset(n_parallel: int = 10, y_horizon: str = "1h", x_horizon: str= "2h"):
    "generate dataset for training."
    import glob
    from pathlib import Path
    cfg = Config()
    x_dates = [Path(_p).stem for _p in glob.glob(os.path.join(cfg.downsampled_10m_dir, "*.parquet"))]
    y_dates = [Path(_p).stem for _p in glob.glob(os.path.join(cfg.ret_1h_dir, "*.parquet"))]
    u_dates = [Path(_p).stem for _p in glob.glob(os.path.join(cfg.univ_dir, "*.parquet"))]
    
    xy_dates = set(x_dates).intersection(set(y_dates)).intersection(set(u_dates))
    
    existing_xy_dates = [Path(_p).stem for _p in glob.glob(os.path.join(cfg.xy_dir, "*.parquet"))]
    typer.echo(f"Total existing dates: {len(existing_xy_dates)}") 
    xy_left_dates = [_d for _d in xy_dates if _d not in existing_xy_dates]  
    typer.echo(f"Total dates to process: {len(xy_left_dates)}")
    
    import ray
    
    @ray.remote(num_cpus=1)
    def run_single(date):
        with pl.StringCache():
            _u = pl.scan_parquet(os.path.join(cfg.univ_dir, f"{date}.parquet")).filter(pl.col('univ_research')).collect()
            symbols = _u.get_column('symbol').unique().to_list()
            _x = (pl.scan_parquet(
                os.path.join(cfg.downsampled_10m_dir, f"{date}.parquet")
                ).cast(
                    {"symbol": pl.Categorical}
                ).filter(
                    pl.col('symbol').is_in(symbols)
                ).sort(
                    ["time", "symbol"]
                ).collect())
            _y = (
                pl.scan_parquet(
                    os.path.join(cfg.ret_1h_dir, f"{date}.parquet")
                    ).cast(
                        {"symbol": pl.Categorical}
                    ).filter(
                        pl.col('symbol').is_in(symbols)
                    ).sort(
                    ["time", "symbol"]
                    ).collect()
                )
            
            window_size = 6 if y_horizon == "1h" else 12
            _x = (
                _x.group_by_dynamic(
                    "time",
                    every=y_horizon,
                    period=x_horizon,
                    label="right",
                    closed="both",
                    group_by="symbol",
                ).agg(
                    *cfg.features_agg,
                    # *[pl.col(yc).last() for yc in self.data_args.y_columns],
                    pl.col("time").first().alias("begin"),
                    pl.col("time").last().alias("end"),
                    pl.col("time").count().alias("count"),
                )
                .filter(pl.col("count") == window_size)
                # .with_columns(*[pl.col(c).list.to_array(6) for c in cols])
            )

            _xy = _x.join(_y, on=[cfg.time_col]+cfg.by_cols, how="left")
            _xy = _xy.with_columns(pl.col(pl.NUMERIC_DTYPES).cast(pl.Float32))
            _xy.write_parquet(os.path.join(cfg.xy_dir, f"{date}.parquet"))
        
    MAX_QUEUE_SIZE = n_parallel
    task_ids = []
    queue_results = []
    for _d in xy_left_dates:
        task_ids.append(run_single.remote(_d))
        if len(task_ids) >= MAX_QUEUE_SIZE:
            done_ids, task_ids = ray.wait(task_ids, num_returns=1)
            queue_results.extend(ray.get(done_ids))
    queue_results.extend(ray.get(task_ids))
    typer.echo("Done")

if __name__ == "__main__":
    app()
