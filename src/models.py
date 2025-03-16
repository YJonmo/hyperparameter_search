import torch

class My_NN_Classifier(torch.nn.Module):
    def __init__(self, in_chan:int=28**2, 
                 num_classes:int=10, 
                 middle_chan:int=20,
                 depth:int=1):
        super().__init__()
        
        self.first_layer = torch.nn.Linear(in_chan, middle_chan)
        self.last_layer = torch.nn.Linear(middle_chan, num_classes)
        self.activation = torch.nn.ReLU()
        self.middle_layers = torch.nn.ModuleList(
            [torch.nn.Sequential(torch.nn.Linear(middle_chan, middle_chan),
                               torch.nn.ReLU()) for _ in range(depth)]
        )

    def forward(self, x) -> torch.Tensor:
        out = self.first_layer(x)
        out = self.activation(out)
        for layer in self.middle_layers:
            out = layer(out)
        out = self.last_layer(out)
        return out
