
import torch
from torch.utils.data import RandomSampler, DataLoader, random_split
from torch import manual_seed 
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from torchvision.transforms import transforms
from torchvision.datasets.mnist import MNIST



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MNIST_Loader:
    """
    A data loader class for the MNIST dataset that handles train-validation splitting and batching.

    Args:
        root_dir (str, optional): Directory to store the dataset. Defaults to './data/'.
        batch_size_train (int, optional): Batch size for training data. Defaults to 100.
        batch_size_val (int, optional): Batch size for validation data. Defaults to 1.
        train_val_split (tuple[int,int], optional): Split sizes for train and validation sets. Defaults to (50000, 10000).
        num_workers (int, optional): Number of workers for data loading. Defaults to 0.
        seed (int, optional): Random seed for reproducibility. Defaults to 10.

    Attributes:
        train_loader (DataLoader): DataLoader for the training dataset
        val_loader (DataLoader): DataLoader for the validation dataset
    """
    def __init__(self, 
                 root_dir:str='./data/',
                 batch_size_train:int=100, batch_size_val:int=1,
                 train_val_split:tuple[int,int]=(50000, 10000),
                 num_workers:int=0,
                 seed:int=10):
        
        manual_seed(seed)
        my_trans = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,)), #for MNIST
                               ])

        dataset_mnist = MNIST(root=root_dir, train=True, transform=my_trans,
                              download=True)

        train, val = random_split(dataset=dataset_mnist, lengths=train_val_split)

        train_sampler= RandomSampler(train)
        val_sampler = RandomSampler(val)
        self.train_loader= DataLoader(dataset=train, sampler=train_sampler,
                                      num_workers=num_workers, batch_size=batch_size_train)
        self.val_loader = DataLoader(dataset=val, sampler=val_sampler,
                                      num_workers=num_workers, batch_size=batch_size_val)
        


def train_loop(model:torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               loader: DataLoader,
               num_epochs: int,
               criterion: torch.nn.Module,
               ) -> list[float, float]:
    """
    Trains a neural network model for the specified number of epochs.

    Args:
        model (torch.nn.Module): The neural network model to train
        optimizer (torch.optim.Optimizer): The optimizer to use for training
        loader (DataLoader): DataLoader containing train and validation data
        num_epochs (int): Number of epochs to train for
        criterion (torch.nn.Module): Loss function to optimize

    Returns:
        tuple: Best validation accuracy and F1 score achieved during training
    """
    model.train()

    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(loader.train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        sum_loss = 0.0
        for i, (X,y) in enumerate(progress_bar):
            X=X.view(-1, 28*28).to(DEVICE)
            y_hat = model(X)
            optimizer.zero_grad()
            loss = criterion(y_hat, y.to(DEVICE))
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            av_loss = sum_loss/(i+1)
            progress_bar.set_postfix({'Loss': f'{av_loss:.4f}'})
        train_loss = sum_loss/(i+1)

        model.eval()
        sum_loss, sum_accuracy, sum_f1, best_accuracy, best_f1 = 0.0, 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for i, (X,y) in enumerate(loader.val_loader):
                X=X.view(-1, 28*28).to(DEVICE)
                pred = model(X)
                loss = criterion(pred, y.to(DEVICE))
                pred = pred.argmax(dim=1).cpu()
                y = y.cpu()
                accuracy = accuracy_score(y_true=y, y_pred=pred)
                f1 = f1_score(y_true=y.cpu(), y_pred=pred, average='macro')
                sum_loss += loss.item()
                sum_accuracy += accuracy
                sum_f1 += f1
                av_loss_val = sum_loss/(i+1)
                av_accuracy = sum_accuracy/(i+1)
                av_f1 = sum_f1/(i+1)
                # using the mean of the two metrics to find the best metric
                if av_accuracy+av_f1 > best_accuracy+best_f1:
                    best_accuracy = av_accuracy
                    best_f1 = av_f1
                
            tqdm.write(f"Epoch: {epoch+1}/{num_epochs}, train loss: {train_loss:.4f}, validation loss: {av_loss_val:0.4f}, accuracy: {av_accuracy:0.4f}, F1 score {av_f1:0.4f}")

    return best_accuracy, best_f1                
