import torch
from torch import nn

EPS = 10**-3


class SudentLoss(nn.Module):
    def __init__(self, gamma=1.0):
        super().__init__()
        self.gamma = gamma
        # self.mse = nn.MSELoss(reduction='sum')

    def forward(self, input, target):
        # mse_loss = self.mse(input, target)
        cauchy_loss = torch.mean(torch.log((input - target)**2 / self.gamma + 1))
        return cauchy_loss


class CorrelationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, input, target):
        X = (input - input.mean()).ravel()
        Y = (target - target.mean()).ravel()
        X_Y = torch.dot(X.T, Y)
        X_X = torch.dot(X.T, X)
        Y_Y = torch.dot(Y.T, Y)
        r_2 = X_Y**2 / (X_X * Y_Y)
        return 1 - r_2


class MedianMSELoss(nn.Module):
    def __init__(self):
        super(MedianMSELoss, self).__init__()

    def forward(self, input, target):
        loss_vector = (input - target)**2
        return torch.median(loss_vector)


class LogisticLoss(nn.Module):
    def __init__(self, s=1.0):
        super().__init__()
        self.s = s

    def forward(self, input, target):
        x = (input - target)
        logistic_loss = torch.mean(x + 2.0 * torch.log(1.0 + torch.exp(-x)))
        return logistic_loss


loss_dict = {
    'normal': nn.MSELoss(),
    'laplace': nn.L1Loss(),
    'student': SudentLoss(),
    'correlation': CorrelationLoss(),
    'median': MedianMSELoss()
#    'logistic': LogisticLoss(),
}
