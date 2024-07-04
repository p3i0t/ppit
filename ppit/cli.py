import datetime
import os
from typing import Any, Dict, Optional

import polars as pl
import ray
import typer

from .config import Config
from .downsample import downsample_1m

app = typer.Typer(
    pretty_exceptions_show_locals=False, 
    pretty_exceptions_enable=False)


def get_or_create_ray_cluster(init_args: Optional[Dict[str, Any]] = None) -> None:
    """
    Check for an existing Ray cluster and connect to it. If none exists, initialize a new one.

    Args:
    init_args (Optional[Dict[str, Any]]): Arguments to pass to ray.init() if a new cluster needs to be created.
                                          Defaults to None, which will use Ray's default settings.

    Returns:
    None
    """
    try:
        # Try to connect to an existing cluster
        ray.init(address='auto', ignore_reinit_error=True)
        print("Connected to existing Ray cluster.")
        cluster_info = ray.cluster_resources()
        print(f"Cluster resources: {cluster_info}")
    except ConnectionError:
        # If no cluster is available, initialize a new one
        print("No existing Ray cluster found. Initializing a new one.")
        init_args = init_args or {}
        ray.init(**init_args)
        print("New Ray cluster initialized.")
        cluster_info = ray.cluster_resources()
        print(f"Cluster resources: {cluster_info}")



def generate_dataset_for_one_day(
    *,
    date: datetime.date,
    u_dir: str,
    x_dir: str,
    y_dir: str,
    ) -> pl.DataFrame:
    cfg = Config()

    dfu = pl.scan_parquet(
        os.path.join(u_dir, f"{date}.parquet")
        ).filter(pl.col('univ_research')).collect()
    dfx = pl.read_parquet(os.path.join(x_dir, f"{date}.parquet"))
    dfx_down = downsample_1m(
            dfx, bars=cfg.features_raw, agg_list=None
        )
    
    dfy = pl.read_parquet(os.path.join(y_dir, f"{date}.parquet"))
    
    df_dataset = dfu.join(dfx_down, on=["date", "symbol"], coalesce=True, how="left")
    df_dataset = df_dataset.join(dfy, on=["date", "symbol"], coalesce=True, how="left")
    return df_dataset

    
    
@app.command()
def down(n_parallel: int = 10, cpu_per_task: int = 4):
    "downsample 1m bars to 10m bars."
    cfg = Config()
    import glob

    get_or_create_ray_cluster(init_args={"num_cpus": n_parallel * cpu_per_task})

    @ray.remote(num_cpus=cpu_per_task)
    def _run_single(src_file: str):
        df = pl.read_parquet(src_file)
        df_down = downsample_1m(
            df, bars=cfg.features_raw, agg_list=None
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

    import glob
    from pathlib import Path

    import ray

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
def gen_crypto_dataset(n_parallel: int = 10, cpu_per_task: int = 4):
    "generate dataset for training."
    import glob
    from pathlib import Path
    
    cfg = Config()
    x_dates = [Path(_p).stem for _p in glob.glob(os.path.join(cfg.bar_1m_dir, "*.parquet"))]
    y_dates = [Path(_p).stem for _p in glob.glob(os.path.join(cfg.ret_1h_dir, "*.parquet"))]
    u_dates = [Path(_p).stem for _p in glob.glob(os.path.join(cfg.univ_dir, "*.parquet"))]
    
    _dates = set(x_dates).intersection(set(y_dates)).intersection(set(u_dates))
    existing_dates = [Path(_p).stem for _p in glob.glob(os.path.join(cfg.research_dataset_dir, "*.parquet"))]
    typer.echo(f"Total existing dates: {len(existing_dates)}") 
    _left_dates = [_d for _d in _dates if _d not in existing_dates]  
    typer.echo(f"Total dates to process: {len(_left_dates)}")
    
    ray_args = {"num_cpus": n_parallel * cpu_per_task}
    get_or_create_ray_cluster(ray_args)
    
    @ray.remote(num_cpus=cpu_per_task)
    def run_single(date: str):
        with pl.StringCache():
            df_dataset = generate_dataset_for_one_day(
                date=datetime.datetime.strptime(date, "%Y-%m-%d").date(),
                u_dir=cfg.univ_dir,
                x_dir=cfg.bar_1m_dir,
                y_dir=cfg.ret_1h_dir,
            )
            if not os.path.exists(cfg.research_dataset_dir):
                os.makedirs(cfg.research_dataset_dir, exist_ok=True)
            df_dataset.write_parquet(
                os.path.join(cfg.research_dataset_dir, f"{date}.parquet")
            )
        
    MAX_QUEUE_SIZE = n_parallel
    task_ids = []
    queue_results = []
    for _d in _left_dates:
        task_ids.append(run_single.remote(_d))
        if len(task_ids) >= MAX_QUEUE_SIZE:
            done_ids, task_ids = ray.wait(task_ids, num_returns=1)
            queue_results.extend(ray.get(done_ids))
    queue_results.extend(ray.get(task_ids))
    typer.echo("Done")

# @click.command()
# @click.option(
#     "--n_jobs", default=10, type=int, help="number of parallel jobs are most."
# )
# @click.option(
#     "--cpus_per_task", "n_cpu", default=2, type=int, help="number of cpus per task."
# )
# @click.option("--verbose", "-v", is_flag=True, help="whether to print progress.")
# def long2widev2(n_jobs, n_cpu, verbose):
#     """Long to wide for v2 bars."""
#     import re
#     import ray
#     import itertools

#     cfg = read_config()
#     dir = Path(cfg.raw.dir)
#     src_dir = dir.joinpath("bar_1m")
#     tgt_dir = dir.joinpath("bar_1m_wide_v2")
#     if verbose is True:
#         click.echo(f"long2wide from {src_dir} to {tgt_dir}")
#     tgt_dir.mkdir(parents=True, exist_ok=True)

#     src_dates = [re.findall(r"\d{4}-\d{2}-\d{2}", d)[0] for d in os.listdir(src_dir)]
#     tgt_dates = [re.findall(r"\d{4}-\d{2}-\d{2}", d)[0] for d in os.listdir(tgt_dir)]
#     trading_dates = sorted(set(src_dates) - set(tgt_dates))
#     if verbose is True:
#         click.echo(f"{trading_dates=}")
#     click.echo(f"Long to wide {len(trading_dates)} tasks to be done.")

#     ray.init(num_cpus=n_jobs * n_cpu, ignore_reinit_error=True, include_dashboard=False)

#     @ray.remote(max_calls=1)
#     def long2wide_fn(date: str):
#         if verbose is True:
#             click.echo(f"running on {date}")
#         df = pl.scan_parquet(f"{src_dir}/{date}.parq")
#         cols = get_bars("v2")
#         df = df.with_columns(pl.col("time").dt.strftime("%H%M").alias("slot"))
#         df = df.with_columns([pl.col(_c).cast(pl.Float32) for _c in cols]).collect()
#         slots = df.get_column("slot").unique().sort().to_list()
#         df = df.pivot(index=["symbol", "date"], columns="slot", values=cols)
#         name_mapping = {
#             f"{col}_slot_{slt}": f"{col}_{slt}"
#             for col, slt in itertools.product(cols, slots)
#         }
#         df = df.rename(name_mapping)
#         df.write_parquet(f"{tgt_dir}/{date}.parq")
#         return date

#     task_ids = []
#     n_task_finished = 0
#     for exp_id, _d in enumerate(trading_dates, 1):
#         if verbose is True:
#             click.echo(f"running on {_d}")
#         task_id = long2wide_fn.options(
#             name="x",
#             num_cpus=n_cpu,
#         ).remote(date=_d)
#         task_ids.append(task_id)

#         if len(task_ids) >= n_jobs:
#             dones, task_ids = ray.wait(task_ids, num_returns=1)
#             ray.get(dones)
#             n_task_finished += 1
#             logger.info(f"{n_task_finished} tasks finished.")
#     ray.get(task_ids)
#     click.echo("task long2wide_v2 done.")




# @click.command()
# def show_config():
#     """Print the configurations in the current config file."""
#     cfg = read_config()
#     click.echo(OmegaConf.to_yaml(cfg))


# @click.command()
# @click.option("--duration", default="30m", type=click.Choice(["15m", "30m", "1h", "2h", '1d', '2d', '5d']))
# def compute_slot_return(duration):
#     """compute returns for 8 intraday slots.
#     """
#     slots = ['0931', '1000', '1030', '1100', '1301', '1330', '1400', '1430']
#     times = [datetime.time(hour=int(s[:2]), minute=int(s[2:])) for s in slots]
#     cfg = read_config()
    
#     price_dir = os.path.join(cfg.raw.dir, 'ohlcv_1m')
#     cols = ['time', 'symbol', 'adj_close']
#     pl.enable_string_cache()
#     df_price = pl.scan_parquet(price_dir + "/*.parq").select(cols).collect()
    
#     df_price_left = df_price.filter(pl.col('time').dt.time().is_in(times))
#     df_adj = adjust_tcalendar_slot_df(duration=duration, start_slot=slots) # two columns: time, next_time
#     end_times = df_adj.select(pl.col('next_time').dt.time().unique()).to_series().to_list()
#     df_price_right = df_price.filter(pl.col('time').dt.time().is_in(end_times))
    
#     df_merge = df_price_left.join(df_adj, on='time')
#     df_merge = df_merge.join(
#         df_price_right, 
#         left_on=['next_time', 'symbol'], 
#         right_on=['time', 'symbol'], 
#         suffix='_right'
#     )
    
#     df_ret = df_merge.select(
#         pl.col('time').cast(pl.Date).alias('date'),
#         pl.col('time'),
#         pl.col('symbol'),
#         pl.col('adj_close_right').truediv(pl.col("adj_close")).sub(1).alias(f'ret_{duration}')
#     )

#     item_dir = os.path.join(cfg.derived.dir, f'ret_{duration}')
#     item_dir = Path(item_dir)
#     item_dir.mkdir(parents=True, exist_ok=True)
    
#     # df_
#     for d, _df in df_ret.partition_by(["date"], as_dict=True).items():
#         # for (d, ), _df in df.partition_by(["date"], as_dict=True).items():
#         _df.write_parquet(f"{item_dir}/{d:%Y-%m-%d}.parq")
#     click.echo("task slot return done.")
    
        
        
if __name__ == "__main__":
    app()
