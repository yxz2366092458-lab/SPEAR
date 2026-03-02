import torch
import torch.nn as nn
import torch.optim as optim
import math

class SelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism for risk assessment
    Captures complex dependencies between different traffic lights
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
            mask: optional attention mask
        Returns:
            attended: [batch_size, seq_len, embed_dim]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections and reshape for multi-head attention
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Q, K, V: [batch_size, num_heads, seq_len, head_dim]
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # scores: [batch_size, num_heads, seq_len, seq_len]
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        # attended: [batch_size, num_heads, seq_len, head_dim]
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Final linear projection
        output = self.out_proj(attended)
        
        return output, attention_weights


class RiskNet(nn.Module):
    """
    Risk Assessment Network with Self-Attention mechanism
    Captures spatial and temporal dependencies for better risk prediction
    """
    def __init__(self, input_dim, hidden_dim=128, num_heads=4, num_layers=2, 
                 num_lights=4, dropout=0.1, lr=1e-4):
        super(RiskNet, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_lights = num_lights
        
        # Calculate features per light (assuming input contains info for all lights)
        self.features_per_light = input_dim // num_lights if num_lights > 0 else input_dim
        
        # Input embedding layer
        self.input_embedding = nn.Linear(self.features_per_light, hidden_dim)
        
        # Positional encoding for spatial relationships between lights
        self.pos_encoding = nn.Parameter(torch.randn(1, num_lights, hidden_dim))
        
        # Self-attention layers
        self.attention_layers = nn.ModuleList([
            SelfAttention(hidden_dim, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Feed-forward network after attention
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Global pooling and risk prediction
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Risk prediction head
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Risk probability [0, 1]
        )
        
        # Auxiliary prediction heads for interpretability
        self.congestion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
    def forward(self, state):
        """
        Compute risk value using self-attention mechanism
        
        Args:
            state: [batch_size, input_dim] - flattened global state
        Returns:
            risk: [batch_size, 1] - overall risk probability
            congestion: [batch_size, 1] - congestion risk
            attention_weights: list of attention matrices from each layer
        """
        batch_size = state.shape[0]
        
        # Reshape state to [batch_size, num_lights, features_per_light]
        state_reshaped = state.view(batch_size, self.num_lights, self.features_per_light)
        
        # Embed input features
        x = self.input_embedding(state_reshaped)  # [batch, num_lights, hidden_dim]
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Apply self-attention layers
        attention_weights_list = []
        for attn_layer, layer_norm in zip(self.attention_layers, self.layer_norms):
            # Self-attention with residual connection
            attended, weights = attn_layer(x)
            attention_weights_list.append(weights)
            x = layer_norm(x + attended)
            
            # Feed-forward with residual
            ffn_out = self.ffn(x)
            x = layer_norm(x + ffn_out)
        
        # Global pooling across lights
        x_pooled = x.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Predict risks
        risk = self.risk_head(x_pooled)  # [batch_size, 1]
        congestion = self.congestion_head(x_pooled)  # [batch_size, 1]
        
        return risk, congestion, attention_weights_list
    
    def get_risk(self, state):
        """Get risk assessment for state (inference mode)"""
        with torch.no_grad():
            risk, _, _ = self.forward(state)
            return risk
    
    def get_attention_map(self, state):
        """Get attention weights for visualization and analysis"""
        with torch.no_grad():
            _, _, attention_weights = self.forward(state)
            return attention_weights
    
    def update(self, state, target_risk, target_congestion=None):
        """
        Update risk network with multi-task learning
        
        Args:
            state: input state
            target_risk: target overall risk
            target_congestion: optional target congestion risk
        """
        predicted_risk, predicted_congestion, _ = self.forward(state)
        
        # Primary risk prediction loss
        risk_loss = self.loss_fn(predicted_risk, target_risk)
        
        # Auxiliary congestion prediction loss (if provided)
        if target_congestion is not None:
            congestion_loss = self.loss_fn(predicted_congestion, target_congestion)
            total_loss = risk_loss + 0.5 * congestion_loss
        else:
            total_loss = risk_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'risk_loss': risk_loss.item(),
            'congestion_loss': congestion_loss.item() if target_congestion is not None else 0
        }
    
    def save(self, path):
        """Save model checkpoint"""
        torch.save({
            'state_dict': self.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'num_lights': self.num_lights
            }
        }, path)
    
    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
