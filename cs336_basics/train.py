from contextlib import contextmanager
import torch
from torch import Tensor
import numpy as np
from numpy import typing as npt
import random
import os
import typing
from pydantic import BaseModel
import logging
from typing import List
import cs336_basics.lego as lego
import time


def get_batch(
    dataset: npt.NDArray, device: str, batch_size: int, context_length: int
) -> tuple[Tensor, Tensor]:
    assert (
        dataset.size >= context_length + 1
    ), f"dataset too small ({dataset.size}) vs context_length ({context_length})"

    sequences = np.empty((batch_size, context_length), dtype=int)
    targets = np.empty((batch_size, context_length), dtype=int)
    for batch in range(batch_size):
        starting = random.randint(0, dataset.size - context_length - 1)
        sequences[batch][:] = dataset[starting : starting + context_length]
        targets[batch][:] = dataset[starting + 1 : starting + context_length + 1]

    sequences_tensor = torch.from_numpy(sequences)
    targets_tensor = torch.from_numpy(targets)

    return (sequences_tensor.to(device=device), targets_tensor.to(device=device))


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    obj = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "iter": iteration,
    }
    torch.save(obj, f=out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    obj = torch.load(f=src)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optim"])
    return obj["iter"]


class LogInterceptor(logging.Filter):
    def __init__(self, intercepting_logging_level=logging.DEBUG):
        super().__init__()
        self.buffer: List[logging.LogRecord] = []
        self._flushing = False
        self.intercepting_logging_level = intercepting_logging_level

    def filter(self, record: logging.LogRecord) -> bool:
        if self._flushing:
            return True

        if record.levelno > self.intercepting_logging_level:
            self.buffer.append(record)
            return False

        return True

    def clear(self):
        self.buffer.clear()

    def flush_to_logger(self, logger: logging.Logger):
        if not self.buffer:
            return

        self._flushing = True
        try:
            for record in self.buffer:
                logger.handle(record)
            self.clear()
        finally:
            self._flushing = False


class TrainingLoopLoggingManager:
    def __init__(
        self,
        logger_name: str,
        initial_step: int = 1,
        every_n_flush: int = 10,
    ):
        self.current_step = initial_step
        self.every_n_flush = every_n_flush
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        self.scoped_filter = LogInterceptor(logging.DEBUG)
        self.logger.addFilter(self.scoped_filter)

    @property
    def log(self):
        return self.logger

    def set_force_flush(self):
        self.force_flush = True

    @contextmanager
    def step_scope(self):
        try:
            yield

            if self.force_flush or (self.current_step - 1) % self.every_n_flush == 0:
                self.force_flush = False
                self.scoped_filter.flush_to_logger(self.logger)
            else:
                self.scoped_filter.clear()
        except Exception as e:
            print(
                f"\n--- [CRASH DETECTED AT STEP {self.current_step}] Flushing Debug Logs ---"
            )
            self.scoped_filter.flush_to_logger(self.logger)
            self.logger.error(f"Encountered exception in step {self.current_step}: {e}")
            raise e


class TrainingConfig(BaseModel):
    name: str

    # Training config
    total_iterations: int
    batch_size: int

    # Model config
    context_length: int
    n_layers: int
    d_model: int
    d_ff: int
    vocab_size: int
    num_heads: int
    theta: float
    device: str
    dtype: str

    # Optimizer config
    adamw_alpha: float
    adamw_beta1: float
    adamw_beta2: float
    adamw_eps: float
    adamw_lambda: float
    adamw_device: str
    adamw_dtype: str

    # Checkpointer config
    checkpoint_every_n: int

    # Logging config
    log_every_n_steps: int = 10

    # Test config
    test_single_sample_overfit: bool = False


class Checkpointer:
    def __init__(self, name: str, checkpoint_every_n: int):
        self.name = name
        self.training_start_time = time.time()
        self.checkpoint_every_n = checkpoint_every_n

    def get_checkpoint_path(
        self,
        step: int,
    ) -> str | os.PathLike | typing.BinaryIO | typing.IO[bytes]:
        return (
            "/tmp"
            / "cs336_llm_checkpoints"
            / self.name
            / f"{self.training_start_time}"
            / step
            / "obj"
        )

    def should_checkpoint(self, step):
        return step % self.checkpoint_every_n == 0


def training_loop(
    dataset_source: npt.NDArray | os.PathLike,
    training_config: TrainingConfig,
):
    tlm = TrainingLoopLoggingManager(
        "training_loop", initial_step=1, every_n_flush=training_config.log_every_n_steps
    )
    tlm.log.info(f"Initializing training loop with TrainingConfig {training_config}")

    model = lego.TransformerModel(
        n_layers=training_config.n_layers,
        d_model=training_config.d_model,
        vocab_size=training_config.vocab_size,
        num_heads=training_config.num_heads,
        d_ff=training_config.d_ff,
        context_length=training_config.context_length,
        theta=training_config.theta,
        device=training_config.device,
        dtype=training_config.dtype,
    )

    optimizer = lego.AdamWOptimizer(
        params=model.parameters,
        alpha=training_config.adamw_alpha,
        beta1=training_config.adamw_beta1,
        beta2=training_config.adamw_beta2,
        epsilon=training_config.adamw_eps,
        lambda_=training_config.adamw_lambda,
        device=training_config.adamw_device,
        dtype=training_config.adamw_dtype,
    )

    checkpointer = Checkpointer(
        name=training_config.name, checkpoint_every_n=training_config.checkpoint_every_n
    )

    sample_tensor, ground_truth_tensor = None, None

    for step in range(1, training_config.total_iterations + 1):
        with tlm.step_scope():
            tlm.log.debug(f"Step {step}: Initializing grad in optim.")
            optimizer.zero_grad()

            tlm.log.debug(f"Step {step}: Sampling a batch of data.")

            if sample_tensor is None or not training_config.test_single_sample_overfit:
                sample_tensor, ground_truth_tensor = get_batch(
                    dataset_source,
                    device=training_config.device,
                    batch_size=training_config.batch_size,
                    context_length=training_config.context_length,
                )

            tlm.log.debug(f"Step {step}: Running forward pass.")
            predictions = model.forward(sample_tensor)
            loss = lego.cross_entropy(predictions, ground_truth_tensor)
            tlm.log.debug(f"Step {step}: Loss: {loss}.")

            tlm.log.debug(f"Step {step}: Running backward pass.")
            loss.backward()

            tlm.log.debug(f"Step {step}: Running optimizer pass.")
            optimizer.step()

            if checkpointer.should_checkpoint(step):
                p = checkpointer.get_checkpoint_path(step)
                tlm.log.info(f"Step {step}: Checkpointing to {p}")
                save_checkpoint(model, optimizer, step, p)
                tlm.log.info(f"Step {step}: Finished checkpointing")
