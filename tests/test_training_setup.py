import pytest
import torch
import torch.nn as nn
from src.losses import InBatchInfoNCELoss
from src.optimizer import build_optimizer, build_scheduler

class DummyTwoTowerModel(nn.Module):
    def __init__(self, input_dim=16, embed_dim=32):
        super().__init__()
        self.user_tower = nn.Linear(input_dim, embed_dim)
        self.item_tower = nn.Linear(input_dim, embed_dim)

    def forward(self, user_features, item_features):
        return self.user_tower(user_features), self.item_tower(item_features)

def test_loss_computation():
    batch_size, dim = 8, 32
    loss_fn = InBatchInfoNCELoss(temperature=0.1)
    q = torch.randn(batch_size, dim)
    k = torch.randn(batch_size, dim)
    loss = loss_fn(q, k)
    assert not torch.isnan(loss)
    assert loss.ndim == 0

def test_learning_rate_warmup_and_step():
    model = nn.Linear(10, 10)
    optimizer = build_optimizer(model, lr=1e-3)
    scheduler = build_scheduler(optimizer, num_warmup_steps=10, num_training_steps=100)
    assert optimizer.param_groups[0]["lr"] == 0.0
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] > 0.0

def test_loss_reduction_on_dummy_batch():
    torch.manual_seed(42)
    batch_size, feature_dim, embed_dim = 16, 16, 32
    model = DummyTwoTowerModel(input_dim=feature_dim, embed_dim=embed_dim)
    loss_fn = InBatchInfoNCELoss(temperature=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    user_inputs = torch.randn(batch_size, feature_dim)
    item_inputs = torch.randn(batch_size, feature_dim)

    u_emb, i_emb = model(user_inputs, item_inputs)
    initial_loss = loss_fn(u_emb, i_emb).item()

    for _ in range(20):
        optimizer.zero_grad()
        u_emb, i_emb = model(user_inputs, item_inputs)
        loss = loss_fn(u_emb, i_emb)
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    assert final_loss < initial_loss, f"Loss failed to decrease: {initial_loss} -> {final_loss}"
