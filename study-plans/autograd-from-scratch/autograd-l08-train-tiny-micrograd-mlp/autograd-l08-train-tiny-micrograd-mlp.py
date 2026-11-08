import torch

def train_tiny_micrograd_mlp(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weights: list[torch.Tensor],
    biases: list[torch.Tensor],
    learning_rate: float,
    steps: int
) -> tuple:
    """
    Returns final predictions, final loss, trained weights, trained biases, and pre-update loss history.
    """
    # Clone parameters to avoid modifying the originals
    W = [w.clone() for w in weights]
    b = [biases[i].clone() for i in range(len(biases))]
    
    pre_update_losses = []
    
    def forward(inputs, W, b):
        activations = [inputs]
        h = inputs
        for l in range(len(W)):
            # H_l = tanh(H_{l-1} @ W_l^T + b_l)
            h = torch.tanh(h @ W[l].T + b[l])
            activations.append(h)
        return h, activations

    # Handle empty batch case: record 0.0 loss for each step
    if inputs.shape[0] == 0:
        preds = torch.zeros(0, dtype=inputs.dtype, device=inputs.device)
        loss = torch.tensor(0.0, dtype=inputs.dtype, device=inputs.device)
        pre_update_losses = [torch.tensor(0.0, dtype=inputs.dtype, device=inputs.device) for _ in range(steps)]
        return preds, loss, W, b, pre_update_losses

    for _ in range(steps):
        # 1. Forward pass
        p, activations = forward(inputs, W, b)
        preds_flat = p.squeeze(-1) # shape (B,)
        
        # 2. Compute loss: L = sum((p_i - y_i)^2)
        diff = preds_flat - targets
        loss = torch.sum(diff ** 2)
        pre_update_losses.append(loss.clone())
        
        # 3. Backward pass (manual gradients)
        grad_h = 2 * diff.unsqueeze(-1)
        
        # Backpropagate through layers in reverse order
        for l in reversed(range(len(W))):
            h_prev = activations[l]
            w_l = W[l]
            
            # Activation is tanh: d(tanh(z))/dz = 1 - tanh^2(z) = 1 - h_curr^2
            h_curr = activations[l + 1]
            dtanh = 1.0 - h_curr ** 2
            
            # Gradient w.r.t pre-activation z_l
            grad_z = grad_h * dtanh
            
            # Gradients for weights and biases
            grad_w = grad_z.T @ h_prev
            grad_b = grad_z.sum(dim=0)
            
            # Gradient for previous layer activations
            if l > 0:
                grad_h = grad_z @ w_l
                
            # 4. Parameter update (Gradient Descent)
            W[l] = W[l] - learning_rate * grad_w
            b[l] = b[l] - learning_rate * grad_b

    # Final forward pass with final parameters
    p, _ = forward(inputs, W, b)
    preds = p.squeeze(-1)
    diff = preds - targets
    final_loss = torch.sum(diff ** 2)

    return preds, final_loss, W, b, pre_update_losses