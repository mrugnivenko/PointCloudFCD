"""This module contains implementation of logger."""

class Logger:
    """Implements logging."""
    
    def __init__(self, experiment) -> None:
        """
        Parameters:
            experiment: Comet.ml experiment.
        """
        self.experiment = experiment
        
    def log_metric(
        self, 
        name: str, 
        value: float, 
        epoch: int
    ) -> None: 
        """
        Log numerical value.
        
        Parameters:
            name: Name of metric;
            value: Value of metric;
            epoch: Epoch number.
        """
        self.experiment.log_metric(name, value, epoch=epoch)
        
    def log_metrics(
        self,
        metrics_names: list, 
        metrics_values: list, 
        epoch: int,
    ) -> None:
        """
        Log list of numerical values.
        
        Parameters:
            metrics_names:
            metrics_values:
            epoch:
        """
        for name, value in zip(metrics_names, metrics_values):
            self.log_metric(name, value, epoch)
            
    def log_train_results(
        self, 
        train_results: list, 
        epoch: int, 
        fold: int,
    ) -> None:
        
        train_results_names = [
            f"train loss (fold #{fold+1})",
            f"train PR-AUC (fold #{fold+1})",
            f"learning rate (fold #{fold+1})",
        ]
        self.log_metrics(train_results_names, train_results, epoch)
        
        
    def log_val_results(
        self, 
        val_results: dict,
        confusion_matrix,
        epoch: int, 
        fold: int,
    ) -> None:
        """
        """
        for name, value in val_results.items():
            self.log_metric(f"test {name} (fold #{fold+1})", value, epoch)
        
        self.experiment.log_confusion_matrix(
            title=f"Test confusion matrix",
            file_name=f"confusion-matrix-fold_{fold+1}-epoch_{epoch}.json",
            matrix=confusion_matrix,
            labels=["No FCD", "FCD"],
        )
