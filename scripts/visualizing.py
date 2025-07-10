import torch
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from typing import List, Tuple
from sklearn.metrics import confusion_matrix


def best_grid(n: int, max_cols: int = 5) -> tuple[int, int]:
    """
    Computes the optimal number of rows and columns to arrange `n` items in a grid.

    This function finds a visually balanced grid layout (rows × columns) for displaying
    `n` items, such as images or plots. It tries to find the smallest grid that fits all
    elements without exceeding `max_cols` columns. If an exact grid match is not found,
    it returns the smallest possible grid that fits all elements within the column limit.

    Args:
        n (int): Total number of items to arrange in a grid.
        max_cols (int, optional): Maximum number of columns allowed in the grid layout. Defaults to 5.

    Returns:
        tuple[int, int]: A tuple (rows, cols) representing the optimal number of rows and columns.
    """

    if n <= max_cols**2:
        for i in range(1, max_cols + 1):
            for j in range(1, max_cols + 1):
                if n == i * j:
                    return i, j
        for i in range(1, max_cols + 1):
            for j in range(1, max_cols + 1):
                if n <= i * j:
                    return i, j
    else:
        j = max_cols
        i = -(-n // j)
        return i, j


def denormalize(tensor:torch.Tensor, mean:list, std:list):
    """
    Reverses normalization of an image tensor for visualization or further processing.

    This function restores an image tensor to its original scale by reversing the normalization
    process applied during data preprocessing. It is typically used to prepare images for
    visualization after normalization with torchvision's `transforms.Normalize`.

    Args:
        tensor (torch.Tensor): Normalized image tensor of shape (C, H, W), where C is the number
            of channels, H is the height, and W is the width. The tensor should have been
            normalized by subtracting a channel-wise mean and dividing by a channel-wise standard
            deviation.
        mean (list): List of per-channel mean values used during normalization. If `None` (along
            with `std`), the function returns a copy of the input tensor without modification.
        std (list): List of per-channel standard deviation values used during normalization. If
            `None` (along with `mean`), the function returns a copy of the input tensor without
            modification.

    Returns:
        torch.Tensor: Denormalized image tensor of the same shape as the input, with pixel values
        rescaled to the original range prior to normalization.

    Raises:
        AssertionError: If `mean` or `std` are not provided as Python lists.

    Notes:
        - The returned tensor is a new copy and does not modify the original tensor in place.
        - The function assumes that the normalization was applied using channel-wise mean and
          standard deviation.
        - The mean and std lists must have lengths equal to the number of channels in the tensor.

    Example:
        >>> img = torch.randn(3, 32, 32)  # Normalized image tensor
        >>> mean = [0.4914, 0.4822, 0.4465]
        >>> std = [0.2470, 0.2435, 0.2616]
        >>> denorm_img = denormalize(img, mean, std)
        >>> plt.imshow(denorm_img.permute(1, 2, 0).numpy())
        >>> plt.show()
    """
    if mean is None and std is None:
        return tensor.clone()

    assert isinstance(mean, list) and isinstance(std, list), 'mean and std must both be lists'

    mean = torch.tensor(mean, device=tensor.device)
    std = torch.tensor(std, device=tensor.device)

    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)

    return tensor * std + mean


def plot_samples_from_dataloader(dataloader: torch.utils.data.DataLoader,
                                 classes: List[str],
                                 *,
                                 mean: list = None,
                                 std: list = None,
                                 R: int = 32,
                                 W: int = 8) -> None:
    """
    Plots image samples from a PyTorch DataLoader with optional denormalization.

    This function displays `R` image samples from the first batch of a provided dataloader.
    The images are arranged in a grid with `W` columns and automatically calculated rows.
    Class labels are displayed above each image. If the dataset was normalized, the function
    can reverse the normalization using the provided `mean` and `std` values for correct visualization.

    Args:
        dataloader (torch.utils.data.DataLoader): PyTorch DataLoader yielding batches of images and labels.
            Images must be of shape (B, C, H, W), where B is batch size, C is the number of channels (typically 3 for RGB).
        classes (List[str]): List of class names corresponding to dataset label indices (e.g., classes[0] is "cat").
        mean (list, optional): Per-channel mean used for normalization. If provided, denormalization will be applied.
            If `mean` and `std` are not provided, no denormalization is applied. Defaults to None.
        std (list, optional): Per-channel standard deviation used for normalization. Must be provided if `mean` is provided.
            Defaults to None.
        R (int, optional): Number of images to display. Must be less than or equal to the batch size. Defaults to 32.
        W (int, optional): Number of columns in the grid. Number of rows is calculated as `R // W`. Defaults to 8.

    Raises:
        ValueError: If `R` is greater than the number of available images in the first batch.
        AssertionError: If `mean` and `std` are partially provided or not in list format (enforced inside `denormalize`).

    Notes:
        - If the dataset was normalized (using transforms.Normalize), denormalization is required for correct visualization.
        - The denormalization is applied using the provided `mean` and `std` per channel.
        - The function assumes labels are provided as integer indices matching the `classes` list.
        - The function directly displays the plot using `matplotlib.pyplot.show()` and does not return any value.

    Example:
        ```python
        plot_samples_from_dataloader(
            dataloader=test_loader,
            classes=['cat', 'dog', 'bird'],
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
            R=16,
            W=4
        )
        # Displays 16 denormalized images in a 4x4 grid.
        ```
    """
    images, labels = next(iter(dataloader))

    if R > len(images):
        raise ValueError(f"Requested {R} images, but batch contains only {len(images)} samples.")

    plt.figure(figsize=(W * 1.5, (R // W) * 1.5), dpi=100)

    for i in range(R):
        plt.subplot(R // W, W, i + 1)

        img_denorm = denormalize(images[i], mean=mean, std=std)
        img = img_denorm.permute(1, 2, 0).numpy()
        label = classes[labels[i].item()]

        plt.imshow(img)
        plt.title(label)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def plot_target_classes_distribution_in_dataset(dataset: torch.utils.data.Dataset, classes: List[str]) -> None:
    """
    Plots the class distribution in a PyTorch dataset.

    This function extracts all target labels from the given dataset, converts them
    to class names using the provided `classes` list, and visualizes the class distribution
    as a bar chart using Seaborn.

    Args:
        dataset (torch.utils.data.Dataset): A PyTorch Dataset object that returns samples in the form
            (image, label) or (x, ..., label), where the label is expected to be the **last element** of the returned tuple.
        classes (List[str]): A list mapping integer label indices to class names. For example, `classes[0] = "cat"`.

    Raises:
        IndexError: If a sample in the dataset does not have the label at the last index.

    Example:
        ```python
        plot_target_classes_distribution_in_dataset(dataset, classes)
        # Displays a countplot showing the number of samples in each class.
        ```
    """

    labels = [classes[dataset[i][-1]] for i in range(len(dataset))]
    df = pd.DataFrame(data=labels, columns=["labels"])

    plt.figure(figsize=(5, 3), dpi=150)
    sns.countplot(data=df, x="labels")
    plt.title("Class Distribution in Dataset")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def plot_losses_and_accuracies(df: pd.DataFrame, figsize: Tuple[int, int] = (8,6), dpi: int = 150):
    """
    Plots training and validation losses and accuracies for multiple models over training epochs.

    This function creates a 2x2 grid of line plots visualizing the progression of:
    - training loss,
    - validation loss,
    - training accuracy,
    - validation accuracy
    for one or more models. The input `DataFrame` is expected to contain per-epoch metrics for each model,
    with model names provided in the 'Model' column.

    Each plot will include separate lines for each model, distinguished by legend labels.

    Args:
        df (pd.DataFrame): DataFrame containing metrics with the following required columns:
            - 'Epoch': number of epoch (int)
            - 'Model': name of the model (str)
            - 'Train Loss': training loss values (float)
            - 'Val Loss': validation loss values (float)
            - 'Train Accuracy': training accuracy values (float in [0, 1])
            - 'Val Accuracy': validation accuracy values (float in [0, 1])
        figsize (tuple, optional): Size of the entire figure in inches (width, height). Defaults to (8, 6).
        dpi (int, optional): Resolution of the figure in dots per inch. Defaults to 150.

    Raises:
        ValueError: If any of the required columns are missing in the input DataFrame.

    Notes:
        - The function uses `seaborn.lineplot()` for styling and smoothing.
        - Accuracy plots are automatically constrained to the [0, 1] range.
        - Metric curves for each model are plotted in the same subplot using legends.

    Example:
        ```python
        df = pd.DataFrame({
            'Model': ['ModelV0'] * 10 + ['ModelV1'] * 10,
            'Train Loss': [...],
            'Val Loss': [...],
            'Train Accuracy': [...],
            'Val Accuracy': [...],
        }, index=[0, 1, ..., 9] * 2)

        plot_losses_and_accuracies(df)
        ```
    """
    x_lim = df['Epoch'].max()
    plt.figure(figsize=figsize, dpi=dpi)
    plt.suptitle("Loss and Accuracy over Epochs", fontsize=14)

    for i in df['Model'].unique():

        df_m = df[df['Model'] == i]

        plt.subplot(2, 2, 1)
        sns.lineplot(data=df_m, x='Epoch', y='Train Loss', label=i)
        plt.title('Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.xlim([0, x_lim])

        plt.subplot(2, 2, 2)
        sns.lineplot(data=df_m, x='Epoch', y='Val Loss', label=i)
        plt.title('Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.xlim([0, x_lim])

        plt.subplot(2, 2, 3)
        sns.lineplot(data=df_m, x='Epoch', y='Train Accuracy', label=i)
        plt.title('Train Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.xlim([0, x_lim])
        plt.ylim([0, 1])

        plt.subplot(2, 2, 4)
        sns.lineplot(data=df_m, x='Epoch', y='Val Accuracy', label=i)
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.xlim([0, x_lim])
        plt.ylim([0, 1])

    plt.tight_layout()
    plt.show()



def find_correct_or_incorrect_samples(model: torch.nn.Module,
                                      dataloader: torch.utils.data.DataLoader,
                                      n: int = 9,
                                      correct: bool = True,
                                      device: str = 'cpu') -> list:
    """
    Extracts a specified number of correctly or incorrectly classified samples from a dataloader.

    This function runs inference on a given dataset and collects `n` samples that were either
    correctly or incorrectly predicted by the model, depending on the `correct` flag.
    Each sample is returned as a tuple of (image_tensor, true_label, predicted_label).

    Args:
        model (torch.nn.Module): Trained PyTorch classification model.
        dataloader (torch.utils.data.DataLoader): Dataloader providing input samples and ground-truth labels.
        n (int, optional): Number of samples to extract. Defaults to 9.
        correct (bool, optional): If True, returns correctly classified samples; if False, returns misclassified ones. Defaults to True.
        device (str, optional): Device on which inference is performed ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        list: A list of `n` tuples, where each tuple is:
            (image_tensor: torch.Tensor, true_label: int, predicted_label: int)

    Notes:
        - This function disables gradient tracking using `torch.inference_mode()`.
        - If fewer than `n` matching samples are found in the dataset, the returned list may be shorter.
    """
    samples = []

    model.eval()
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            y_label = torch.argmax(y_pred, dim=1)

            for img, lbl, pred in zip(X, y, y_label):
                if len(samples) >= n:
                    break
                if (lbl == pred) == correct:
                    samples.append((img, lbl, pred))

    return samples


def plot_correct_or_incorrect_samples(model: torch.nn.Module,
                                      dataloader: torch.utils.data.DataLoader,
                                      classes: List[str],
                                      *,
                                      mean: list = None,
                                      std: list = None,
                                      n: int = 9,
                                      correct: bool = True,
                                      dpi: int = 100,
                                      device: str = 'cpu') -> None:
    """
    Displays a grid of correctly or incorrectly classified image samples from a model with optional denormalization.

    This function visualizes a specified number (`n`) of either correctly or incorrectly classified samples
    from a given dataloader. The images are displayed in a grid layout with class labels shown above each image.
    Correct predictions are displayed in green. Incorrect predictions display both true and predicted labels in red.
    If the dataset was normalized, the function can reverse the normalization using the provided `mean` and `std`
    for proper visualization.

    Args:
        model (torch.nn.Module): Trained PyTorch classification model.
        dataloader (torch.utils.data.DataLoader): DataLoader providing evaluation data (images and labels).
        classes (List[str]): List of class names corresponding to dataset label indices.
        mean (list, optional): Per-channel mean used for normalization. If provided, denormalization will be applied.
            If `mean` and `std` are not provided, no denormalization is applied. Defaults to None.
        std (list, optional): Per-channel standard deviation used for normalization. Must be provided if `mean` is provided.
            Defaults to None.
        n (int, optional): Number of samples to display. Defaults to 9.
        correct (bool, optional): If True, displays correctly classified samples. If False, displays misclassified samples.
            Defaults to True.
        dpi (int, optional): DPI setting for the output figure. Controls figure resolution. Defaults to 100.
        device (str, optional): Computation device for inference ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        None: This function directly displays a matplotlib figure and does not return any value.

    Raises:
        AssertionError: If `mean` and `std` are partially provided or not in list format (enforced inside `denormalize`).

    Notes:
        - The function uses `find_correct_or_incorrect_samples` to extract the desired samples based on model predictions.
        - The grid layout is automatically determined using `best_grid(n)`.
        - The images are assumed to be in CHW format and are converted to HWC for display with `matplotlib`.
        - If the dataset was normalized during preprocessing, providing `mean` and `std` is required for proper visualization.
        - The function does not shuffle the dataset; it returns samples in the order found during dataloader iteration.

    Example:
        ```python
        plot_correct_or_incorrect_samples(
            model=trained_model,
            dataloader=test_loader,
            classes=['cat', 'dog', 'bird'],
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
            n=12,
            correct=False,
            dpi=120,
            device='cuda'
        )
        # Displays 12 incorrectly classified, denormalized images in a grid layout.
        ```
    """
    samples = find_correct_or_incorrect_samples(model=model, dataloader=dataloader, n=n, correct=correct, device=device)

    n_rows, n_cols = best_grid(n)
    w, h = n_cols * 2, n_rows * 2

    plt.figure(figsize=(w, h), dpi=dpi)
    plt.suptitle(f"{'C' if correct else 'Inc'}orrectly predicted samples", fontsize=14)

    for i, (img, lbl, pred) in enumerate(samples):
        plt.subplot(n_rows, n_cols, i + 1)

        img_denorm = denormalize(img, mean=mean, std=std)
        img = img_denorm.permute(1, 2, 0).numpy()

        plt.imshow(img)

        if correct:
            plt.title(f"{classes[pred]}", color="green", fontsize=11)
        else:
            plt.title(f"True: {classes[lbl]} | Pred: {classes[pred]}", color="red", fontsize=10)

        plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model: torch.nn.Module,
                          dataloader: torch.utils.data.DataLoader,
                          target_classes: list[str],
                          *,
                          normalize: str = 'true',
                          annot: bool = True,
                          annot_size: int = 16,
                          fmt: str = '.2f',
                          cbar: bool = True,
                          cmap: str = 'Blues',
                          device: str = 'cpu') -> None:
    """
    Plots a confusion matrix heatmap for a classification model.

    This function computes predictions on a given dataloader and visualizes
    the confusion matrix using a seaborn heatmap. Supports various formatting
    and display options, including normalization, annotation font size, and color maps.

    Args:
        model (torch.nn.Module): Trained model for generating predictions.
        dataloader (torch.utils.data.DataLoader): DataLoader providing evaluation data.
        target_classes (list[str]): List of class names corresponding to model output indices.
        normalize (str, optional): Normalization strategy to apply ('true', 'pred', 'all', or None). Defaults to 'true'.
        annot (bool, optional): Whether to annotate each cell with its numeric value. Defaults to True.
        annot_size (int, optional): Font size for cell annotations. Defaults to 16.
        fmt (str, optional): Format string for annotations. Defaults to '.2f'.
        cbar (bool, optional): Whether to display the color bar. Defaults to True.
        cmap (str, optional): Color map to use for the heatmap. Defaults to 'Blues'.
        device (str, optional): Computation device ('cpu' or 'cuda'). Defaults to 'cpu'.

    Raises:
        AssertionError: If the target_classes list contains non-unique elements.

    Returns:
        None. Displays the confusion matrix using matplotlib.
    """
    assert len(target_classes) == len(set(target_classes)), \
        "target_classes must contain unique class names."

    y_true = torch.tensor([], dtype=torch.int64).to(device)
    y_pred = torch.tensor([], dtype=torch.int64).to(device)

    model.eval()
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_label = torch.argmax(model(X), dim=1)
            y_true = torch.cat([y_true, y], dim=0)
            y_pred = torch.cat([y_pred, y_label], dim=0)

        cm = confusion_matrix(
            y_true.cpu(),
            y_pred.cpu(),
            labels=range(len(target_classes)),
            normalize=normalize,
        )

        cm_df = pd.DataFrame(
            cm,
            columns=target_classes,
            index=target_classes,
        )

    sns.heatmap(
        data=cm_df,
        annot=annot,
        cbar=cbar,
        cmap=cmap,
        fmt=fmt,
        annot_kws={'size': annot_size},
        vmin=0,
        vmax=1,
    )
    plt.title('Confusion Matrix', fontsize=18)
    plt.ylabel('True labels', fontsize=14)
    plt.xlabel('Predicted labels', fontsize=14)
