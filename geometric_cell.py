import torch
from utils import device

class GeometricCell:
    def __init__(self, dimensions, value=0.0):
        self.dimensions = dimensions
        self.position = torch.zeros(dimensions, device=device)
        self.value = torch.tensor(value, device=device)
        self.bias = 0.01
        self.connections = []
        self.gradient = torch.zeros(dimensions, device=device)
        self.momentum = 0.8

    def update_value(self, new_value):
        l2_reg = 0.005 * torch.sum(self.position ** 2)  # Reduzido
        self.value = new_value * (1 - self.momentum) + self.value * self.momentum - l2_reg

    def add_connection(self, other_cell):
        self.connections.append(other_cell)