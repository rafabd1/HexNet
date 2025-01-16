import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import itertools

from shape_types import GeometricShape

class NetworkVisualizer:
    def __init__(self):
        self.fig = None
        self.ax = None
        self.cluster_ax = None
        self.scatter = None
        self.lines = []
        self.paused = False
        self.colorbar = None
        
    def setup(self):
        plt.ion()
        self.fig = plt.figure(figsize=(18, 8))
        gs = self.fig.add_gridspec(1, 3, width_ratios=[2, 1, 0.1])
        
        self.ax = self.fig.add_subplot(gs[0], projection='3d')
        self.ax.set_title("Topologia Geométrica")
        
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_zlim(-1.5, 1.5)
        self.ax.view_init(elev=20, azim=45)
        
        self.cluster_ax = self.fig.add_subplot(gs[1])
        self.cluster_ax.set_title("Clusters")
        
        self.cax = self.fig.add_subplot(gs[2])
        
        return self
        
    def _get_geometric_positions(self, network):
        # Distribuição mais uniforme usando distribuição esférica
        n = len(network.cells)
        indices = torch.arange(0, n, dtype=torch.float)
        phi = torch.arccos(1 - 2 * indices / n)  # Distribuição uniforme no eixo z
        theta = torch.pi * (1 + 5**0.5) * indices  # Ângulo áureo para melhor distribuição
        
        r = 1.0
        x = r * torch.sin(phi) * torch.cos(theta)
        y = r * torch.sin(phi) * torch.sin(theta)
        z = r * torch.cos(phi)
        
        return torch.stack([x, y, z], dim=1)
    
    def update(self, network):
        if self.fig is None:
            self.setup()
            
        if self.paused:
            return
            
        if hasattr(self, 'colorbar') and self.colorbar is not None:
            self.cax.clear()
            
        self.cluster_ax.clear()
        self.lines = []
        
        positions = torch.stack([cell.position for cell in network.cells])
        values = torch.stack([c.value for c in network.cells])
        values = torch.mean(values, dim=1)
        
        geometric_pos = self._get_geometric_positions(network)
        x, y, z = geometric_pos[:, 0], geometric_pos[:, 1], geometric_pos[:, 2]
        values_np = values.cpu().numpy()
        
        scatter = self.ax.scatter(
            x, y, z,
            c=values_np,
            cmap='viridis',
            s=200
        )
        
        if network.shape_type == GeometricShape.HYPERCUBE:
            for i in range(len(x)):
                for j in range(i+1, len(x)):
                    diff = sum([abs(x[i]-x[j]), abs(y[i]-y[j]), abs(z[i]-z[j])])
                    if diff <= 2.0:  
                        self.ax.plot(
                            [x[i], x[j]],
                            [y[i], y[j]],
                            [z[i], z[j]],
                            color='gray',
                            alpha=0.3,
                            linewidth=2
                        )
        else:
            for i, cell in enumerate(network.cells):
                for conn in cell.connections:
                    j = network.cells.index(conn)
                    strength = abs(values_np[i] - values_np[j])
                    self.ax.plot(
                        [x[i], x[j]],
                        [y[i], y[j]],
                        [z[i], z[j]],
                        color='gray',
                        alpha=0.3,
                        linewidth=1 + strength
                    )
        
        self.ax.view_init(elev=30, azim=45)
        self.ax.set_box_aspect([1,1,1])
        
        from sklearn.cluster import DBSCAN
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=2)
        pos_2d = pca.fit_transform(positions.cpu().numpy())
        
        clustering = DBSCAN(eps=0.3, min_samples=2).fit(pos_2d)
        labels = clustering.labels_
        
        self.cluster_ax.scatter(
            pos_2d[:, 0], pos_2d[:, 1],
            c=labels,
            cmap='tab20',
            s=100,
            alpha=0.6
        )
        
        unique_labels = set(labels)
        for label in unique_labels:
            if label != -1:
                mask = labels == label
                center = pos_2d[mask].mean(axis=0)
                self.cluster_ax.annotate(
                    f'C{label}',
                    center,
                    xytext=(5, 5),
                    textcoords='offset points'
                )
        
        title = (
            f'Células: {len(network.cells)} | '
            f'Clusters: {len(unique_labels)-1 if -1 in labels else len(unique_labels)}'
        )
        self.fig.suptitle(title)
        
        self.colorbar = plt.colorbar(scatter, cax=self.cax, label='Nível de Ativação')
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Save the figure
        self.fig.savefig('./assets/network_visualization.png')
