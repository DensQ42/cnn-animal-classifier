import time
import os
import torch
from typing import List, Tuple, Optional
from tqdm.auto import tqdm


class EarlyStopping:
    """
    Implements early stopping logic for model training.

    This class monitors the validation loss over epochs and triggers
    early stopping if no sufficient improvement is observed for a specified number
    of consecutive epochs.

    Attributes:
        patience (int): Number of epochs to wait for improvement before stopping.
        min_delta (float): Minimum change in validation loss to qualify as improvement.
        early_stop (bool): Whether training should be stopped early.
        best_loss (float): Best validation loss seen so far.
        counter (int): Number of consecutive epochs without improvement.
    """

    def __init__(self, patience: int = 20, min_delta: float = 0.00001):
        """
        Args:
            patience (int): Number of epochs to wait for improvement.
            min_delta (float): Minimum change to qualify as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.reset()

    def __call__(self, test_loss: float):
        """
        Updates the early stopping state based on the current validation loss.

        Args:
            test_loss (float): Current epoch's validation loss.

        Notes:
            If the validation loss improves more than `min_delta`, the counter resets.
            Otherwise, the counter increases. If it reaches `patience`, early stopping is triggered.
        """
        if test_loss < self.best_loss - self.min_delta:
            self.best_loss = test_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def reset(self):
        """
        Resets the early stopping state.
        Useful if the same EarlyStopping instance is reused across multiple training runs.
        """
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: str = 'cpu') -> Tuple[float, float]:
    """
    Executes a single training epoch over the provided DataLoader.

    This function performs a full training pass using the given model, loss function,
    and optimizer. After each batch, gradients are computed and model parameters are updated.
    The function computes the average loss and accuracy across all training samples in this epoch.

    Args:
        model (torch.nn.Module): The model to be trained.
        dataloader (torch.utils.data.DataLoader): Dataloader providing the training batches.
        loss_fn (torch.nn.Module): Loss function to optimize.
        optimizer (torch.optim.Optimizer): Optimizer that updates model parameters.
        device (str, optional): Device on which computations will be performed ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        Tuple[float, float]: A tuple containing:
            - average loss across all batches in this epoch,
            - overall accuracy (correct / total samples).

    Notes:
        - This function computes the average of the **per-batch** loss values.
        - The average is taken over the number of batches, not the number of individual samples.
        - If batch sizes vary and precise loss per sample is desired, you may need a weighted approach.
        - Gradient clipping with max_norm=5 is applied to prevent exploding gradients.
    """
    avg_loss = 0
    avg_accuracy = 0
    total_samples = 0

    model.train()

    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)

        y_pred = model(X)

        avg_accuracy += (torch.argmax(y_pred, dim=1) == y).sum().item()
        total_samples += y.size(0)

        loss = loss_fn(y_pred, y)
        avg_loss += loss.item() * y.size(0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

    avg_loss /= total_samples
    avg_accuracy /= total_samples

    return avg_loss, avg_accuracy


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: str = 'cpu') -> Tuple[float, float]:
    """
    Evaluates the model on a test or validation set for a single epoch.

    This function disables gradient computation and switches the model to evaluation mode
    using `model.eval()` and `torch.inference_mode()`. It  computes the average loss and
    accuracy across all samples in the given DataLoader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): Dataloader providing the evaluation data.
        loss_fn (torch.nn.Module): Loss function used for evaluation.
        device (str, optional): Device to run computations on ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        Tuple[float, float]: A tuple containing:
            - average loss across all batches in this epoch,
            - overall accuracy (correct / total samples).

    Notes:
        - This function computes the average of the **per-batch** loss values.
        - The average is taken over the number of batches, not the number of individual samples.
        - If batch sizes vary and precise loss per sample is desired, you may need a weighted approach.
        - No parameter updates or gradient calculations are performed during this step.
    """
    avg_loss = 0
    avg_accuracy = 0
    total_samples = 0

    model.eval()

    with torch.inference_mode():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            y_pred = model(X)

            avg_accuracy += (torch.argmax(y_pred, dim=1) == y).sum().item()
            total_samples += y.size(0)

            loss = loss_fn(y_pred, y)
            avg_loss += loss.item() * y.size(0)

    avg_loss /= total_samples
    avg_accuracy /= total_samples

    return avg_loss, avg_accuracy


def training_loop(model: torch.nn.Module,
                  train_dataloader: torch.utils.data.DataLoader,
                  test_dataloader: torch.utils.data.DataLoader,
                  loss_fn: torch.nn.Module,
                  optimizer: torch.optim.Optimizer,
                  num_epochs: int,
                  scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                  log_every_n_epochs: int = None,
                  early_stopping: Optional[EarlyStopping] = None,
                  model_name: str = None,
                  device: str = 'cpu',
) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    """
    Executes the training and evaluation loop for a PyTorch model over multiple epochs.

    This function iteratively trains and evaluates the given model using provided data loaders.
    It tracks losses, accuracies, and epoch durations. Optional features include learning rate scheduling,
    early stopping, periodic logging, and automatic saving of the best model (based on validation loss).

    Args:
        model (torch.nn.Module): The PyTorch model to be trained and evaluated.
        train_dataloader (torch.utils.data.DataLoader): Dataloader providing training batches.
        test_dataloader (torch.utils.data.DataLoader): Dataloader providing test or validation batches.
        loss_fn (torch.nn.Module): Loss function used to calculate training and evaluation loss.
        optimizer (torch.optim.Optimizer): Optimizer used to update the model's parameters.
        num_epochs (int): Total number of training epochs.
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler or torch.optim.lr_scheduler.ReduceLROnPlateau], optional):
            Learning rate scheduler. If provided, it will be stepped after each epoch. For schedulers of type
            `ReduceLROnPlateau`, the validation loss is passed as an argument to `step()`. For others, `step()` is called without arguments.
            Defaults to None.
        log_every_n_epochs (int, optional): Interval (in epochs) at which training progress is printed.
            If None, no logging is performed. Defaults to None.
        early_stopping (Optional[EarlyStopping], optional): An instance of EarlyStopping. If provided,
            enables early stopping based on validation loss. Defaults to None.
        model_name (str, optional): Custom name to use when saving the best model. If not provided,
            the model’s class name is used. Defaults to None.
        device (str, optional): Device on which to perform computations ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        Tuple[List[float], List[float], List[float], List[float], List[float]]:
            A tuple containing:
                - train_losses (List[float]): Average training loss per epoch.
                - test_losses (List[float]): Average test/validation loss per epoch.
                - train_accuracies (List[float]): Training accuracy per epoch.
                - test_accuracies (List[float]): Test/validation accuracy per epoch.
                - epoch_times (List[float]): Duration of each epoch in seconds.

    Raises:
        AssertionError: If `num_epochs` is not a positive integer.
        AssertionError: If `log_every_n_epochs` is provided but is not a positive integer.

    Notes:
        - Uses `train_step` and `test_step` for per-epoch training and validation.
        - If a learning rate scheduler is provided, it is updated at the end of each epoch.
          For `ReduceLROnPlateau`, `scheduler.step(val_loss)` is called.
        - Logs learning rate value when logging is enabled.
        - If `early_stopping` is triggered, training terminates before reaching `num_epochs`.
        - The best model (based on minimum validation loss) is saved to `models/` as a `.pth` file.
        - The model must implement a `.save(path)` method for saving.
    """
    assert log_every_n_epochs is None or (log_every_n_epochs > 0 and isinstance(log_every_n_epochs, int)), \
        'log_every_n_epochs must be an integer number greater than zero'
    assert num_epochs > 0 and isinstance(num_epochs, int), \
        'epochs must be an integer number greater than zero'

    epoch_pad = len(str(num_epochs+1))

    train_losses = []
    test_losses = []
    epoch_times = []
    train_accuracies = []
    test_accuracies = []
    best_test_loss = float('inf')

    os.makedirs('models', exist_ok=True)

    for epoch in tqdm(range(num_epochs)):

        start = time.perf_counter()

        train_loss, train_accuracy = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        test_loss, test_accuracy = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
        )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
            else:
                scheduler.step()

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        if log_every_n_epochs is not None and (epoch % log_every_n_epochs) == 0:
            print(f'Epoch: {epoch+1:<{epoch_pad}} | ', end='')
            print(f'Train loss: {round(train_loss, 4):<6} | ', end='')
            print(f'Test loss: {round(test_loss, 4):<6} | ', end='')
            print(f'Train accuracy: {round(train_accuracy, 4):<6} | ', end='')
            print(f'Test accuracy: {round(test_accuracy, 4):<6} | ', end='')
            if scheduler is not None:
                current_lr = optimizer.param_groups[0]["lr"]
                print(f'LR: {current_lr:.8f}', end='')
            print()

        if early_stopping is not None:
            early_stopping(test_loss)
            if early_stopping.early_stop:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break

        if best_test_loss > test_loss:
            best_test_loss = test_loss
            if model_name is None:
                model_name = model.__class__.__name__
            save_path = f'models/{model_name}_best.pth'
            model.save(save_path)

        duration = time.perf_counter() - start
        epoch_times.append(duration)

    return train_losses, test_losses, train_accuracies, test_accuracies, epoch_times
