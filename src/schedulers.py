import numpy as np


class Scheduler:
    def __init__(self, initial_lr: float = 0.01):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.name = "BaseScheduler"

    def get_lr(self, epoch: int, **kwargs) -> float:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.name}(initial_lr={self.initial_lr})"


class PowerScheduler(Scheduler):
    def __init__(self, initial_lr: float = 0.01, step: int = 100, power: float = 1.0):
        super().__init__(initial_lr)
        self.step = step
        self.power = power
        self.name = "Power"

    def get_lr(self, epoch: int, **kwargs) -> float:
        self.current_lr = self.initial_lr / (1 + epoch / self.step) ** self.power
        return self.current_lr


class ExponentialScheduler(Scheduler):
    def __init__(self, initial_lr: float = 0.01, decay_step: int = 100):
        super().__init__(initial_lr)
        self.decay_step = decay_step
        self.name = "Exponential"

    def get_lr(self, epoch: int, **kwargs) -> float:
        self.current_lr = self.initial_lr * (0.1 ** (epoch / self.decay_step))
        return self.current_lr


class PiecewiseScheduler(Scheduler):
    def __init__(
        self, initial_lr: float = 0.01, drop_step: int = 50, drop_factor: float = 0.1
    ):
        super().__init__(initial_lr)
        self.drop_step = drop_step
        self.drop_factor = drop_factor
        self.name = "Piecewise"

    def get_lr(self, epoch: int, **kwargs) -> float:
        self.current_lr = self.initial_lr * (
            self.drop_factor ** (epoch // self.drop_step)
        )
        return self.current_lr


class OneCycleScheduler(Scheduler):
    def __init__(
        self,
        initial_lr: float = 0.001,
        max_lr: float = 0.01,
        total_epochs: int = 100,
        pct_start: float = 0.3,
    ):
        super().__init__(initial_lr)
        self.max_lr = max_lr
        self.total_epochs = total_epochs
        self.pct_start = pct_start
        self.name = "1Cycle"

    def get_lr(self, epoch: int, **kwargs) -> float:
        cycle = int(self.total_epochs * self.pct_start)

        if epoch < cycle:
            self.current_lr = self.initial_lr + (self.max_lr - self.initial_lr) * (
                epoch / cycle
            )
        else:
            remaining = self.total_epochs - cycle
            self.current_lr = self.max_lr - (self.max_lr - self.initial_lr) * (
                (epoch - cycle) / remaining
            )

        return self.current_lr


class ReduceOnPlateauScheduler(Scheduler):
    def __init__(
        self,
        initial_lr: float = 0.01,
        factor: float = 0.5,
        patience: int = 10,
        min_lr: float = 1e-6,
    ):
        super().__init__(initial_lr)
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr

        self.best_loss = float("inf")
        self.counter = 0
        self.name = "ReduceOnPlateau"

    def get_lr(self, epoch: int, **kwargs) -> float:
        loss = kwargs.get("loss", None)

        if loss is None:
            return self.current_lr

        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.current_lr = max(self.current_lr * self.factor, self.min_lr)
                self.counter = 0

                print(
                    f"ReduceOnPlateau: LR decreased to {self.current_lr:.6f} at epoch {epoch}"
                )

        return self.current_lr
