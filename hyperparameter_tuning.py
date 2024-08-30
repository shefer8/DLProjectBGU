import matplotlib.pyplot as plt
# import optuna



# Find the optimal number os Epocs:
def plot_epoch_loss(training_losses, validation_losses=None):
    plt.figure(figsize=(10, 6))
    plt.plot(training_losses, label='Training Loss')
    if validation_losses:
        plt.plot(validation_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_metrics(history, metric_name='accuracy'):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history[metric_name], label=f'Training {metric_name.capitalize()}')
    plt.plot(history.history[f'val_{metric_name}'], label=f'Validation {metric_name.capitalize()}')
    plt.title(f'{metric_name.capitalize()} over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name.capitalize())
    plt.legend()
    plt.grid(True)
    plt.show()




