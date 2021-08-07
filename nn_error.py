# pytorch-lightning
from pytorch_lightning import LightningModule, Trainer, metrics
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import fetch_california_housing, load_boston, fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from distribution import dist_dict
from sklearn.utils import warnings
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning
from collections import OrderedDict
from losses import loss_dict
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


class NNError(LightningModule):
    def __init__(self, train_dataset, error='gaussian'):
        super().__init__()
        self.train_dataset = train_dataset
        batch_size = self.train_dataset.get_input_shape() // 10
        self._set_hparams({'batch_size': batch_size, 'learning_rate': 5e-3})
        input_dim = self.train_dataset.get_input_dim()
        W = input_dim // 2
        self.layers = nn.Sequential(nn.Linear(input_dim, W),
                                    nn.ELU(inplace=True),
                                    nn.Linear(W, 1)
                                    )
        self.loss = loss_dict[error]
        self.mse = metrics.MeanSquaredError()

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.05)
        return [optimizer], [scheduler]

    def forward(self, x):
        y_hat = self.layers(x)
        return y_hat

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': self.loss(y_hat, y)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.layers(x)
        loss = self.loss(y_hat, y)

        mse = self.mse(y_hat, y)
        output = OrderedDict({
            'loss': loss,
            'train_mse': mse,  # everything must be a tensor
        })
        return output

    def test_epoch_end(self, outputs):
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        results = {'log': {'avg_test_loss': avg_loss, 'avg_acc': avg_acc}}
        return results


class CustomDataset(Dataset):
    def __init__(self, data, target):
        self.input_dim = data.shape[1]
        self.input_shape = data.shape[0]
        self.samples = [(torch.from_numpy(d).float(), torch.from_numpy(t).float()) for d, t in zip(data, target)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_input_dim(self):
        return self.input_dim

    def get_input_shape(self):
        return self.input_shape


if __name__ == '__main__':
    # Breast cancer dataset
    data_scaler = StandardScaler()
    target_scaler = PowerTransformer()
    # target_scaler = StandardScaler()  # PowerTransformer()
    # data, target = load_boston(return_X_y=True)
    data, target = fetch_california_housing(return_X_y=True)
    input_dim = data.shape[1]
    target = target.reshape(-1, 1)
    data_scaled = data_scaler.fit_transform(data)
    target_scaled = target_scaler.fit_transform(target)

    # Plot
    bin_number = 25
    plt.figure(1)
    plt.hist(target, bins=bin_number)
    plt.title('Target')
    plt.savefig('images/target.png')
    plt.close(1)
    plt.figure(1)
    plt.hist(target_scaled, bins=bin_number)
    plt.title('Target scaled')
    plt.savefig('images/target_scaled.png')
    plt.close(1)
    # raise ValueError('stop')

    data_train, data_test, target_train, target_test = \
        train_test_split(data_scaled, target_scaled, test_size=0.20)

    # Dataset
    train_dataset = CustomDataset(data_train, target_train)
    val_dataset = CustomDataset(data_test, target_test)
    # train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, num_workers=4)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=32, num_workers=4)

    # model_correlation = NNError(error='cauchy', train_dataset=train_dataset)
    # trainer = Trainer(max_epochs=1)  # , gpus=-1)  # For GPUs
    # trainer.fit(model=model_correlation, val_dataloaders=val_dataloader)

    for fig, error_name in enumerate(loss_dict.keys()):
        print(error_name)
        model = NNError(error=error_name, train_dataset=train_dataset)
        trainer = Trainer(max_epochs=10)  # , gpus=-1)  # For GPUs
        trainer.fit(model=model, val_dataloaders=val_dataloader)
        pred_scaled = model.forward(torch.from_numpy(data_train).float()).detach().numpy()
        # error = pred_scaled - target_train
        error = target_scaler.inverse_transform(pred_scaled) - target_scaler.inverse_transform(target_train)
        error_mean = np.mean(error)
        error_std = error.std()
        x = np.linspace(- 3 * error_std, 3 * error_std, bin_number)
        y = [dist_dict[error_name](x_n, 0.0, error.std()) for x_n in x]
        # mse_error = (error**2).mean()
        plt.figure(fig)
        plt.hist(error, density=True, bins=bin_number)
        plt.plot(x, y, 'k')
        plt.title(error_name + ' mean =' + str(error_mean))
        # plt.show()
        plt.savefig('images/' + error_name)
        plt.close(fig)
        print()
