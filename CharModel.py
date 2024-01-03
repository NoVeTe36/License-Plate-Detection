from torch import nn

class CharModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.stack = nn.Sequential(
            nn.Conv2d(1, 5, 3, 1, "same"),
            nn.ReLU(),
            nn.BatchNorm2d(5),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(5, 10, 3, 1, "same"),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(10*25*25, 200),
            nn.Dropout(0.2),
            nn.Linear(200, 32),
            nn.Softmax(1),
        )

    def forward(self, X):
        X = self.stack(X)
        return X 