import torch
import torch.nn as nn
import torch.optim as optim

class RiskNet(nn.Module):
    """
    Risk Assessment Network for evaluating state-action risk
    """
    def __init__(self, input_dim, hidden_dims=[128, 64], lr=1e-4):
        super(RiskNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Output risk probability [0, 1]
        
        self.network = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
    def forward(self, state):
        """
        Compute risk value for given state
        Args:
            state: [batch_size, input_dim]
        Returns:
            risk: [batch_size, 1] - risk probability
        """
        return self.network(state)
    
    def get_risk(self, state):
        """Get risk assessment for state"""
        with torch.no_grad():
            return self.forward(state)
    
    def update(self, state, target_risk):
        """Update risk network"""
        predicted = self.forward(state)
        loss = self.loss_fn(predicted, target_risk)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
