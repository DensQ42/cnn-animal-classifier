import torch
from sklearn.metrics import f1_score

def calculate_f1_score(model: torch.nn.Module,
                       dataloader: torch.utils.data.DataLoader,
                       average: str = 'macro',
                       device: str = 'cpu') -> float:
    
    """
    Calculates the F1 score for a given model on a provided dataset.

    This function evaluates the model on the given dataloader (e.g. test or validation),
    collects all predicted and true class labels, and computes the F1 score.

    Args:
        model (torch.nn.Module): The trained PyTorch model to evaluate.
        dataloader (torch.utils.data.DataLoader): The DataLoader containing evaluation data.
        average (str, optional): Type of averaging performed on the data.
            Can be 'micro', 'macro', 'weighted', or 'none'. Default is 'macro'.
        device (str, optional): Device on which to run the model ('cpu' or 'cuda'). Default is 'cpu'.

    Returns:
        float: The computed F1 score.

    Raises:
        ValueError: If `average` is not a supported type.
        ValueError: If no matching labels are found for computing F1.
    """

    assert average in {'micro', 'macro', 'weighted', 'none'}, \
        f"Invalid average type: {average}. Choose from 'micro', 'macro', 'weighted', or 'none'."

    y_true = torch.tensor([], dtype=torch.int64).to(device)
    y_pred = torch.tensor([], dtype=torch.int64).to(device)

    model.eval()
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            preds = torch.argmax(model(X), dim=1)
            y_true = torch.cat([y_true, y], dim=0)
            y_pred = torch.cat([y_pred, preds], dim=0)

    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    try:
        f1 = f1_score(y_true_np, y_pred_np, average=average)
    except ValueError as e:
        raise ValueError(f"F1-score calculation failed: {e}")

    return f1
