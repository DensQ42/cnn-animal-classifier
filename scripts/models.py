import torch

class ModelUtilsMixin:
    """
    A utility mixin for PyTorch models that provides convenient methods
    for parameter counting, saving/loading weights, inference, freezing,
    and summarizing the model.
    """

    def count_params(self) -> int:
        """
        Count the number of trainable parameters in the model.

        Returns:
            int: Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str) -> None:
        """
        Save the model's state_dict to a file.

        Args:
            path (str): Path to save the weights.
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: str = 'cpu') -> None:
        """
        Load model weights from a file.

        Args:
            path (str): Path to the saved weights.
            device (str): Target device to load weights onto. Default is 'cpu'.
        """
        self.load_state_dict(torch.load(path, map_location=device))
        self.eval()

    @torch.no_grad()
    def predict(self, x: torch.Tensor, batch_size: int = None, device: str = 'cpu') -> torch.Tensor:
        """
        Run inference on input tensor. Supports batch-wise prediction.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W) or similar.
            batch_size (int, optional): If provided, splits input into batches.
            device (str): Device to run inference on ('cpu' or 'cuda').

        Returns:
            torch.Tensor: Predictions from the model, moved to CPU.
        """
        self.eval()
        self.to(device)

        if batch_size is None:
            pred = self(x.to(device))
            return pred.cpu()

        preds = []
        for i in range(0, len(x), batch_size):
            batch = x[i:i + batch_size].to(device)
            pred = self(batch)
            preds.append(pred.cpu())

        return torch.cat(preds, dim=0)

    def freeze(self) -> None:
        """
        Freeze all model parameters to disable gradient computation.
        Useful during feature extraction or evaluation.
        """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """
        Unfreeze all model parameters to enable training.
        """
        for param in self.parameters():
            param.requires_grad = True

    def __str__(self) -> str:
        """
        Return a readable string representation of the model class and structure.

        Returns:
            str: Formatted model description.
        """
        return f"{self.__class__.__name__}(\n{super().__str__()})"



class RandomGuessModel(torch.nn.Module):
    """
    A PyTorch model that generates random predictions, independent of input data.

    This model serves as a random baseline for classification tasks. Given a batch of inputs,
    it returns random scores for each class. The outputs are unnormalized random values and do not
    represent probabilities. This model is useful for comparison with real models to verify that a
    trained model performs better than random guessing.

    Attributes:
        num_classes (int): The number of target classes.

    Args:
        num_classes (int): The number of classes in the classification task.

    Example:
        ```python
        model = RandomGuessModel(num_classes=10)
        x = torch.randn(32, 3, 32, 32)  # Example input batch
        output = model(x)
        print(output.shape)  # torch.Size([32, 10])
        ```
    """

    def __init__(self, num_classes: int):
        """
        Initializes the RandomGuessModel.

        Args:
            num_classes (int): The number of classes in the classification task.
        """
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates random, unnormalized predictions for each input sample.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ...). The actual content of x is ignored.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, num_classes) containing random values
            uniformly sampled from the range [0, 1). The outputs are not normalized and do not sum to 1.

        Notes:
            - The outputs are not logits from a learned model; they are pure random noise.
            - This model is intended to serve as a baseline for random guessing performance.
            - The outputs can be directly used with loss functions like `torch.nn.CrossEntropyLoss`,
              which apply softmax internally.
        """
        batch_size = x.shape[0]
        return torch.rand(batch_size, self.num_classes, device=x.device)
