"""
Fixed Neural Network Components for SEIR DAD
Using the exact LazyDelta implementation from death process
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro.distributions import Delta


class LazyDelta(Delta):
    """Lazy evaluation wrapper - exact copy from death process"""
    
    def __init__(self, fn, prototype, log_density=0.0, event_dim=0, validate_args=None):
        self.fn = fn
        super().__init__(
            prototype,
            log_density=log_density,
            event_dim=event_dim,
            validate_args=validate_args,
        )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LazyDelta, _instance)
        new.fn = self.fn
        batch_shape = torch.Size(batch_shape)
        new.v = self.v.expand(batch_shape + self.event_shape)
        new.log_density = self.log_density.expand(batch_shape)
        super(Delta, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=torch.Size()):
        # The shape of self.v will have expanded along with any .expand calls
        shape = sample_shape + self.v.shape
        output = self.fn()
        return output.expand(shape)

    @property
    def variance(self):
        return torch.zeros_like(self.v)

    def log_prob(self, x):
        return self.log_density


class EncoderNetwork(nn.Module):
    """Encoder network matching death process structure exactly"""
    
    def __init__(
        self,
        design_dim=1,
        observation_dim=1,
        hidden_dim=128,
        encoding_dim=16,
        n_hidden_layers=2,
        activation=nn.Softplus,
    ):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.activation_layer = activation()
        
        # Input layer
        self.input_layer = nn.Linear(design_dim + observation_dim, hidden_dim)
        
        # Hidden layers
        if n_hidden_layers > 1:
            self.middle = nn.Sequential(
                *[
                    nn.Sequential(nn.Linear(hidden_dim, hidden_dim), activation())
                    for _ in range(n_hidden_layers - 1)
                ]
            )
        else:
            self.middle = nn.Identity()
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, encoding_dim)

    def forward(self, xi, y, **kwargs):
        # Handle scalar inputs
        if not isinstance(xi, torch.Tensor):
            xi = torch.tensor([xi], dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor([y], dtype=torch.float32)
        
        # Ensure proper dimensions
        if xi.dim() == 0:
            xi = xi.unsqueeze(0)
        if y.dim() == 0:
            y = y.unsqueeze(0)
        
        # Stack inputs (matching death process)
        inputs = torch.stack([xi.squeeze(), y.squeeze()], dim=-1)
        
        # Forward pass
        x = self.input_layer(inputs)
        x = self.activation_layer(x)
        x = self.middle(x)
        x = self.output_layer(x)
        
        return x


class EmitterNetwork(nn.Module):
    """Emitter network matching death process structure"""
    
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        n_hidden_layers=2,
        activation=nn.Softplus,
    ):
        super().__init__()
        self.activation_layer = activation()
        
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        if n_hidden_layers > 1:
            self.middle = nn.Sequential(
                *[
                    nn.Sequential(nn.Linear(hidden_dim, hidden_dim), activation())
                    for _ in range(n_hidden_layers - 1)
                ]
            )
        else:
            self.middle = nn.Identity()
            
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, r):
        x = self.input_layer(r)
        x = self.activation_layer(x)
        x = self.middle(x)
        x = self.output_layer(x)
        return x


class SetEquivariantDesignNetwork(nn.Module):
    """Main DAD network - exactly matching death process"""
    
    def __init__(self, encoder, emitter, empty_value):
        super().__init__()
        self.encoder = encoder
        self.emitter = emitter if emitter is not None else nn.Identity()
        self.register_buffer("prototype", empty_value.clone())
        self.register_parameter("empty_value", nn.Parameter(empty_value))

    def lazy(self, *design_obs_pairs):
        def delayed_function():
            return self.forward(*design_obs_pairs)

        lazy_delta = LazyDelta(
            delayed_function, self.prototype, event_dim=self.prototype.dim()
        )
        return lazy_delta

    def forward(self, *design_obs_pairs):
        if len(design_obs_pairs) == 0:
            # For efficiency: learn the first design separately
            output = self.empty_value
        else:
            # Encode all history and sum
            sum_encoding = sum(
                self.encoder(design, obs)
                for design, obs in design_obs_pairs
            )
            output = self.emitter(sum_encoding)
        
        return output


class BatchDesignBaseline(nn.Module):
    """Static baseline that learns T fixed designs"""
    
    def __init__(self, T, design_dim, output_activation=nn.Identity()):
        super().__init__()
        self.register_buffer("prototype", torch.zeros(design_dim))
        
        # Initialize designs
        design_init = torch.distributions.Normal(0, 0.5)
        self.designs = nn.ParameterList(
            [
                nn.Parameter(design_init.sample(torch.zeros(design_dim).shape))
                for i in range(T)
            ]
        )
        self.output_activation = output_activation

    def lazy(self, *design_obs_pairs):
        def delayed_function():
            return self.forward(*design_obs_pairs)

        lazy_delta = LazyDelta(
            delayed_function, self.prototype, event_dim=self.prototype.dim()
        )
        return lazy_delta

    def forward(self, *design_obs_pairs):
        j = len(design_obs_pairs)
        return self.output_activation(self.designs[j])


def create_seir_dad_network(
    hidden_dim=128,
    encoding_dim=16,
    n_hidden_layers=2,
    activation='softplus'
):
    """
    Factory function to create SEIR DAD network
    Matching death process architecture exactly
    """
    
    # Select activation
    activations = {
        'softplus': nn.Softplus,
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'elu': nn.ELU
    }
    
    activation_fn = activations.get(activation, nn.Softplus)
    
    # Create encoder
    encoder = EncoderNetwork(
        design_dim=1,
        observation_dim=1,
        hidden_dim=hidden_dim,
        encoding_dim=encoding_dim,
        n_hidden_layers=n_hidden_layers,
        activation=activation_fn
    )
    
    # Create emitter
    emitter = EmitterNetwork(
        input_dim=encoding_dim,
        hidden_dim=hidden_dim,
        output_dim=1,
        n_hidden_layers=n_hidden_layers,
        activation=activation_fn
    )
    
    # Create full network
    design_net = SetEquivariantDesignNetwork(
        encoder=encoder,
        emitter=emitter,
        empty_value=torch.ones(1)  # Initial design value
    )
    
    return design_net


if __name__ == "__main__":
    # Test the networks
    print("Testing Fixed SEIR DAD Networks")
    print("-" * 40)
    
    # Create network
    design_net = create_seir_dad_network()
    
    # Test with no history
    design_0 = design_net()
    print(f"Initial design (no history): {design_0.item():.4f}")
    
    # Test with some history
    xi_1 = torch.tensor(5.0)
    y_1 = torch.tensor(10.0)
    
    design_1 = design_net((xi_1, y_1))
    print(f"Design after 1 observation: {design_1.item():.4f}")
    
    # Test lazy evaluation
    lazy_design = design_net.lazy()
    sampled = lazy_design.rsample()
    print(f"Lazy sampled design: {sampled.item():.4f}")
    
    # Test with batch
    batch_shape = torch.Size([3])
    expanded = lazy_design.expand(batch_shape)
    batch_sample = expanded.rsample()
    print(f"Batch sample shape: {batch_sample.shape}")
    
    print("\nAll network tests passed!")