import torch
import torch.nn.functional as F


class hERGClassifier(torch.nn.Module):
    """
    hERG Classifier architecture to process fingerprints.

    Parameters:
        inputSize : int
            The input size of the fingerprints representations.
        output_dim : int
            The output dimension of the classifier.
    """

    def __init__(self, input_size, output_size, dropout_rate=0.2):
        super(hERGClassifier, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, 200, bias=True)
        torch.nn.init.kaiming_normal_(self.linear1.weight, nonlinearity="relu")
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        #self.bn1 = torch.nn.BatchNorm1d(num_features=200)

        self.linear2 = torch.nn.Linear(200, 200, bias=True)
        torch.nn.init.kaiming_normal_(self.linear2.weight, nonlinearity="relu")
        self.dropout2 = torch.nn.Dropout(dropout_rate)
        #self.bn2 = torch.nn.BatchNorm1d(num_features=200)

        self.linear3 = torch.nn.Linear(200, output_size, bias=True)
        torch.nn.init.kaiming_normal_(self.linear3.weight, nonlinearity="relu")

    def forward(self, x):
        out = self.dropout1(F.relu(self.linear1(x)))
        out = self.dropout2(F.relu(self.linear2(out)))
        out = F.softmax(self.linear3(out), dim=1)
        return out

    def save(self, path):
        """
        Save model with its parameters to the given path.
        Conventionally the path should end with "*.model".

        Inputs:
        - path: path string
        """
        print("Saving model... %s" % path)
        torch.save(self.state_dict(), path)

    def load(self, path):
        """
        Load model dictionary. The
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        device = torch.device("cpu")
        self.load_state_dict(torch.load(path, map_location=device))
