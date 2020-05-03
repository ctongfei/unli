import torch
import torch.nn


class MLP(torch.nn.Sequential):

    def __init__(self, *args):
        super(MLP, self).__init__(*args)
        self.apply(self.init_weights)

    def init_weights(self, module):
        pass
        # if isinstance(module, torch.nn.Linear) and module.bias is not None:
        #     module.bias.data.zero_()
        #     module.weight.data.normal_(mean=0.0, std=0.1)  # from BERT

    @classmethod
    def from_params(cls, params):
        hidden_sizes = params["hidden_sizes"]
        activation = {
            "none": lambda: torch.nn.Sequential(),
            "relu": lambda: torch.nn.ReLU(),
            "tanh": lambda: torch.nn.Tanh(),
            "sigmoid": lambda: torch.nn.Sigmoid()
        }[params["activation"]]

        n = len(hidden_sizes)
        layers = []
        for i in range(n - 2):
            layers.append(
                torch.nn.Linear(
                    in_features=hidden_sizes[i],
                    out_features=hidden_sizes[i + 1]
                )
            )
            layers.append(activation())

        if n > 1:
            layers.append(
                torch.nn.Linear(
                    in_features=hidden_sizes[n - 2],
                    out_features=hidden_sizes[n - 1]
                )
            )

        layers.append(
            {
                None: lambda: torch.nn.Sequential(),
                "none": lambda: torch.nn.Sequential(),
                "sigmoid": lambda: torch.nn.Sigmoid()
            }[params.get("last", None)]()
        )

        mlp = cls(*layers)

        return mlp
