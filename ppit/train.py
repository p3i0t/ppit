from typing import Literal, Iterable, Optional, Any
from collections import namedtuple, defaultdict
from functools import lru_cache

from pydantic.dataclasses import dataclass
import torch
from torch.optim import AdamW
import polars as pl
import numpy as np

from datasource import CryptoDataSource, CryptoDataArguments


@dataclass(kw_only=True)
class TrainArguments:
    """Model training arguments."""

    model: Literal["GPT", "LSTM"] = "GPT"
    d_in: int
    d_out: int
    device: Literal["cpu", "cuda"] = "cuda"
    epochs: int = 20
    train_batch_size: int = 1024
    eval_batch_size: int = 1024
    test_batch_size: int = 2000
    lr: float = 5e-5
    weight_decay: float = 1e-3
    patience: int = 6
    seed: int = 42
    watch_metric: str = "loss"  # metric name on validation metrics.
    watch_mode: Literal["min", "max"] = "min"
    dataloader_drop_last: bool = False


BatchData = namedtuple("BatchData", ["time", "symbol", "x", "y", "y_columns"])


class Trainer:
    """
    Trainer class.

    ...
    Attributes
    ----------
    model: nn.Module
            The model to be trained.
    args: TrainArguments
            The arguments to be used for training.
    train_dataset: StockDataset
            The training dataset.
    eval_dataset: StockDataset
            The evaluation dataset.

    Methods
    -------
    get_eval_dataloader(eval_dataset: StockDataset = None) -> Iterable[BatchData]
            Get the evaluation dataloader.
    get_train_dataloader() -> Iterable[BatchData]
            Get the training dataloader.
    create_optimizer()
            Create the optimizer.
    train_epoch()
            Train the model for one epoch.
    eval_epoch(dataloader: Iterable[BatchData]) -> dict[str, Any]
            Evaluate the model for one epoch.
    _process_batch(batch: BatchData) -> BatchData
            Process the batch.
    train()
            Train the model.
    save_checkpoint(save_dir: str)
            Save the checkpoint.
    """

    def __init__(
        self,
        *,
        args: TrainArguments,
        train_dataset: StockDataset = None,
        eval_dataset: StockDataset = None,
        # compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
    ):
        self.args = args
        self.model = get_model(name=args.model, d_in=args.d_in, d_out=args.d_out).to(
            self.args.device
        )
        # model.train()
        if self.model is None:
            raise ValueError("Trainer: model cannot be None.")
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.optimizer = AdamW(
            self.model.parameters(), lr=self.args.lr, weight_decay=1e-3
        )
        # self.compute_metrics = compute_metrics  # TODO

    def get_eval_dataloader(
        self, eval_dataset: Optional[StockDataset] = None
    ) -> Iterable[BatchData]:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return NumpyStockDataLoader(
            dataset=eval_dataset,
            shuffle=False,
            batch_size=self.args.eval_batch_size,
            drop_last=self.args.dataloader_drop_last,
            # device=self.args.device,
        )

    def get_train_dataloader(self) -> Iterable[BatchData]:
        """
        Get the training dataloader, i.e. transforming the training dataset into batches.

        Returns:
                Iterator[BatchData]: The training dataloader.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires an eval dataset.")
        return NumpyStockDataLoader(
            dataset=self.train_dataset,
            shuffle=True,
            batch_size=self.args.train_batch_size,
            drop_last=self.args.dataloader_drop_last,
            # device=self.args.device,
        )

    def train_epoch(self):
        self.model.train()
        loss_list = []
        loader = self.get_train_dataloader()
        for batch_idx, batch in enumerate(loader):
            x, y = self._process_xy(batch.x, batch.y)
            pred: torch.Tensor = self.model(x)[:, self.args.output_indices, :]
            loss = (pred - y).pow(2).mean()
            loss_list.append(loss.detach().cpu().item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        loss = np.mean(loss_list)
        return loss

    def eval_epoch(
        self, dataloader: Iterable[BatchData], prediction_loss_only: bool = False
    ) -> dict[str, Any]:
        self.model.eval()

        res_dict = defaultdict(list)
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                x, y = self._process_xy(batch.x, batch.y)
                pred: torch.Tensor = self.model(x)[:, self.args.output_indices, :]
                loss = (pred - y).pow(2).mean()
                res_dict["loss"].append(loss.detach().cpu().item())

                if not prediction_loss_only:  # todo: modify when y is not single slot
                    res_dict["date"].append(batch.date)
                    res_dict["symbol"].append(batch.symbol)
                    pred_cols = [f"pred_{c}" for c in batch.y_columns]
                    df_pred = pl.DataFrame(
                        pred.detach().cpu().flatten(start_dim=1).numpy(),
                        schema=pred_cols,
                    )
                    res_dict["pred"].append(df_pred)
                    df_y = pl.DataFrame(
                        y.detach().cpu().flatten(start_dim=1).numpy(),
                        schema=batch.y_columns,
                    )
                    res_dict["y"].append(df_y)

            output = {"loss": np.mean(res_dict["loss"])}
            if not prediction_loss_only:
                df_ds = pl.DataFrame(
                    {
                        "date": np.concatenate(res_dict["date"]),
                        "symbol": np.concatenate(res_dict["symbol"]),
                    }
                )

                df_pred = pl.concat(res_dict["pred"])
                df_y = pl.concat(res_dict["y"])
                df = df_ds.hstack(df_pred).hstack(df_y)
                ic = df.group_by("date").agg(
                    pl.corr(a, b) for a, b in zip(df_y.columns, df_pred.columns)
                )
                ic = ic.select(df_y.columns).mean()

                ic_dict = {y: ic.get_column(y).item() for y in df_y.columns}
                del df, df_pred, df_y, df_ds, ic
            else:
                ic_dict = {}

            return output | ic_dict

    def _process_xy(self, x, y=None):
        """
        Process the batch, i.e. move the batch to the device and convert the batch to tensor.
        Args:
                batch (BatchData): The stock batch to be processed.

        Returns:
                BatchData: The processed batch.
        """
        x = torch.Tensor(x).to(self.args.device, non_blocking=True)
        x = torch.nan_to_num(x, nan=0.0)
        if y is not None:
            y = torch.Tensor(y).to(self.args.device, non_blocking=True)
            y = torch.nan_to_num(y, nan=0.0)
            return x, y
        else:
            return x

    def train(self):
        args = self.args
        if args.monitor_mode == "min":
            best = float("inf")
        else:
            best = float("-inf")
        best_epoch = 0
        best_state = None

        for epoch in range(int(self.args.epochs)):
            train_loss = self.train_epoch()
            eval_dict = self.eval_epoch(
                self.get_eval_dataloader(), prediction_loss_only=True
            )
            logger.info(f"======> Epoch {epoch+1:03d}")
            logger.info(f"train_loss: {train_loss:.4f}")
            # logger.info("Evaluation:")
            for k, v in eval_dict.items():
                logger.info(f"eval, {k}: {v:.4f}")

            if (
                args.monitor_mode == "max" and eval_dict[args.monitor_metric] > best
            ) or (args.monitor_mode == "min" and eval_dict[args.monitor_metric] < best):
                logger.info("======> New Optimal")
                best = eval_dict[args.monitor_metric]
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                self._save_checkpoint(str(args.milestone_dir))
            # logger.info(f"Saved checkpoint to {args.save_dir}.")

            if epoch - best_epoch >= args.patience:
                logger.info("======> Early stop.")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def _save_checkpoint(self, save_dir: str) -> None:
        """
        Save the checkpoint, including model state, training arguments.
        Args:
            save_dir: output directory of this trainer.
        Returns:

        """
        checkpoint_dir = f"{save_dir}/{CHECHPOINT_META.prefix_dir}"
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        torch.save(
            self.model.state_dict(),
            f"{checkpoint_dir}/{CHECHPOINT_META.model}",
        )
        import yaml

        with open(f"{checkpoint_dir}/{CHECHPOINT_META.training_args}", "w") as f:
            # f.write(self.args.model_dump())
            yaml.dump(self.args.model_dump(), f)
        # torch.save(self.args, f"{checkpoint_dir}/{CHECHPOINT_META.training_args}")

    def evaluate(self, eval_dataset: Optional[StockDataset] = None) -> dict[str, Any]:
        """
        Evaluate the model on the given dataset.

        Parameters
        ----------
        eval_dataset: StockDataset
                The evaluation dataset.

        Returns
        -------
        dict[str, Any]
                The evaluation results.
        """
        loader = self.get_eval_dataloader(eval_dataset)
        return self.eval_epoch(loader)


class TrainPipeline:
    def __init__(
        self, data_args: CryptoDataArguments, training_args: TrainingArguments
    ):
        self.data_args = data_args
        self.datasource = CryptoDataSource(data_args)
        self.training_args = training_args

    def forward(self):
        datasets = self.datasource()
        # print(f"Training pipeline with args: {self.args}")

    def __call__(self):
        self.forward()
