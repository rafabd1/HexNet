import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import torch
import itertools
from enum import Enum
from geometric_cell import GeometricCell
from utils import device
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

class GeometricShape(Enum):
    HYPERCUBE = "hypercube"
    DODECAHEDRON = "dodecahedron"
    TESSERACT = "tesseract"
    ADAPTIVE = "adaptive"

class ComplexGeometricNetwork:
    def __init__(self, input_dimensions, output_dimensions, shape_type=GeometricShape.HYPERCUBE):
        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions
        self.dimensions = input_dimensions
        self.shape_type = shape_type
        self.n_cells = 64  # Mais células
        self.hidden_dim = 128  # Dimensão intermediária
        self.skip_alpha = 0.3  # Peso menor skip connection
        self.diversity_weight = 0.5  # Aumentado
        self.cells = []
        self.topology = None
        self.training_errors = []
        self.validation_errors = []
        self.best_error = float('inf')
        self.stress_threshold = 0.3
        self.learning_rate = 0.01  # Aumentado
        self.min_lr = 0.001  # Aumentado
        self.scheduler_patience = 15  # Mais paciência 
        self.scheduler_factor = 0.9  # Mais suave
        self.weight_decay = 1e-6  # Reduzido
        self.scheduler_patience = 10  # Aumentado
        self.attention_weights = None
        self.skip_connections = True
        self.activation = lambda x: torch.relu(x)
        self.momentum = 0.85
        self.patience = 40
        self.min_delta = 0.0005
        self.morph_strength = 2.0  # Aumentado
        self._initialize_topology()

    def _initialize_topology(self):
        if self.shape_type in {GeometricShape.HYPERCUBE, GeometricShape.DODECAHEDRON, GeometricShape.TESSERACT}:
            self._create_hypercube_topology()
        else:
            self._create_hypercube_topology()

    def _create_hypercube_topology(self):
        if self.shape_type == GeometricShape.HYPERCUBE:
            max_dims = min(self.dimensions, 4)
            base_vertices = list(itertools.product([0, 1], repeat=max_dims))
            vertices = torch.tensor(base_vertices, dtype=torch.float, device=device)
            if max_dims < self.dimensions:
                padding = torch.zeros(len(vertices), self.dimensions - max_dims, device=device)
                vertices = torch.cat((vertices, padding), dim=1)

            print(f"Creating {len(vertices)} cells with {self.dimensions} dimensions")
            self.cells = [GeometricCell(self.dimensions) for _ in range(len(vertices))]
            for i, cell in enumerate(self.cells):
                cell.position = vertices[i].clone()
                for j, other_cell in enumerate(self.cells):
                    diff = torch.sum(torch.abs(vertices[i] - vertices[j]))
                    if diff.item() == 1.0:
                        cell.add_connection(other_cell)
            self.topology = {'vertices': vertices, 'cells': self.cells}
        else:
            # Inicialização mais diversa
            std = torch.sqrt(torch.tensor(2.0 / (self.input_dimensions + self.hidden_dim), device=device))
            self.cells = []
            
            # Cria células com posições mais diversas
            positions = []  # Lista para armazenar posições
            for i in range(self.n_cells):
                cell = GeometricCell(self.hidden_dim)
                angle = torch.tensor(2 * torch.pi * i / self.n_cells, device=device)
                position = torch.zeros(self.hidden_dim, device=device)
                position[0] = torch.cos(angle) 
                position[1] = torch.sin(angle)
                position[2:] = torch.randn(self.hidden_dim-2, device=device) * std
                cell.position = position
                self.cells.append(cell)
                positions.append(position)
            
            # Conexões baseadas em similaridade cosine
            self._update_all_connections()
            
            # Cria topologia com as posições geradas
            self.topology = {
                'vertices': torch.stack(positions),
                'cells': self.cells
            }
    
    def _update_all_connections(self):
        """Atualiza conexões entre células baseado em similaridade cosseno"""
        # Obtém todas posições
        positions = torch.stack([cell.position for cell in self.cells])
        
        # Calcula similaridade cosseno entre todas células
        norm = torch.norm(positions, dim=1, keepdim=True)
        normalized = positions / (norm + 1e-8)
        similarity = torch.mm(normalized, normalized.t())
        
        # Limpa conexões existentes
        for cell in self.cells:
            cell.connections = []
        
        # Conecta células mais similares
        k = min(4, len(self.cells)-1)  # Número de conexões por célula
        _, indices = torch.topk(similarity, k=k+1, dim=1)
        
        for i, cell in enumerate(self.cells):
            # Ignora primeira posição (é a própria célula)
            for j in indices[i, 1:]:
                other_cell = self.cells[j]
                if other_cell not in cell.connections:
                    cell.add_connection(other_cell)
                    other_cell.add_connection(cell)

    def _map_input_to_cells(self, batch_data):
        # Verifica e adapta dimensões
        batch_size = batch_data.shape[0]
        input_size = int(np.sqrt(batch_data.shape[1]))
        
        try:
            # Tenta reshape para formato de imagem
            x = batch_data.view(batch_size, 1, input_size, input_size)
        except:
            # Fallback se dimensões não forem quadradas
            x = batch_data.view(batch_size, 1, -1, 1)
            x = F.interpolate(x, size=(32, 32))
        
        # Multi-scale feature extraction
        x1 = F.avg_pool2d(x, kernel_size=4)  # 8x8
        x2 = F.max_pool2d(x, kernel_size=4)  # 8x8
        x3 = F.avg_pool2d(x, kernel_size=8)  # 4x4
        
        # Concatena features
        features = torch.cat([
            x1.flatten(1),
            x2.flatten(1),
            x3.flatten(1)
        ], dim=1)
        
        # Projection layer
        if not hasattr(self, 'projection'):
            in_features = features.shape[1]
            self.projection = torch.nn.Linear(in_features, self.hidden_dim).to(device)
        
        # Normaliza e projeta
        features = self.projection(features)
        features = F.layer_norm(features, features.shape[1:])
        
        # Atualiza células
        cell_positions = torch.stack([c.position for c in self.cells])
        similarity = torch.matmul(features, cell_positions.t())
        attention = torch.softmax(
            similarity / torch.sqrt(torch.tensor(self.hidden_dim, device=device)) * 2.0,  # temperatura aumentada
            dim=1
        )
        
        for i, cell in enumerate(self.cells):
            weighted_input = torch.sum(features * attention[:, i].unsqueeze(1), dim=0)
            cell.update_value(weighted_input)

    def process_input_batch(self, batch_data):
        batch_size = batch_data.shape[0]
        all_preds = []
    
        for i in range(batch_size):
            single_input = torch.tensor(batch_data[i], dtype=torch.float, device=device).unsqueeze(0)
            self._map_input_to_cells(single_input)
            self._morph_structure(morph_factor=0.1)
    
            values = torch.stack([c.value for c in self.cells], dim=0)
            values = self._attention_layer(values)
            feature_map = self._reduce_dimensions(values).unsqueeze(0)
    
            if self.skip_connections:
                # Extrai features do input com dimensões corretas
                input_features = single_input[:, :self.output_dimensions]
                
                # Garante dimensionalidade correta
                if not hasattr(self, 'skip_projection'):
                    input_dim = input_features.shape[1]
                    self.skip_projection = torch.nn.Sequential(
                        torch.nn.Linear(input_dim, self.output_dimensions),
                        torch.nn.LayerNorm(self.output_dimensions)
                    ).to(device)
                
                # Projeta input
                skip_features = self.skip_projection(input_features)
                
                # Gate adaptativo
                gate = torch.sigmoid(
                    torch.sum(feature_map * skip_features, dim=-1, keepdim=True)
                )
                
                # Combina features
                feature_map = feature_map * (1 - gate) + skip_features * gate
    
            feature_map = torch.layer_norm(feature_map, feature_map.shape)
            preds = torch.softmax(feature_map / 0.5, dim=-1)
            all_preds.append(preds)
    
        return torch.cat(all_preds, dim=0)

    def _morph_structure(self, morph_factor):
        if self.shape_type == GeometricShape.ADAPTIVE:
            # Força maior nas transformações
            morph_factor *= 2.0
            
            # Calcula centróides por classe
            values = torch.stack([c.value for c in self.cells])
            positions = torch.stack([c.position for c in self.cells])
            
            # Agrupa células por similaridade
            clusters = self._cluster_cells(positions, values)
            
            # Aplica forças mantendo diversidade
            for cell, cluster_idx in zip(self.cells, clusters):
                same_cluster = positions[clusters == cluster_idx]
                other_clusters = positions[clusters != cluster_idx]
                
                # Força de atração para mesmo cluster
                attract = torch.mean(same_cluster - cell.position, dim=0)
                
                # Força de repulsão de outros clusters
                repel = torch.mean(cell.position - other_clusters, dim=0)
                
                # Aplica forças
                force = (attract * morph_factor) + (repel * self.diversity_weight)
                cell.position = cell.position + force

    def _cluster_cells(self, positions, values):
        from sklearn.cluster import KMeans
        k = self.output_dimensions
        kmeans = KMeans(n_clusters=k)
        return kmeans.fit_predict(positions.detach().cpu().numpy())

    def _attention_layer(self, x):
        if self.attention_weights is None or self.attention_weights.shape[0] != x.shape[0]:
            self.attention_weights = torch.ones(x.shape[0], device=device) / x.shape[0]
        attention = torch.softmax(self.attention_weights * torch.mean(x), dim=0)
        return x * attention.unsqueeze(-1) if x.dim() > 1 else x * attention

    def _reduce_dimensions(self, values):
        n_cells = len(self.cells)
        region_size = max(1, n_cells // self.output_dimensions)
        
        outputs = []
        for i in range(self.output_dimensions):
            start_idx = i * region_size
            end_idx = min(start_idx + region_size, n_cells)
            region_values = values[start_idx:end_idx]
            mean_val = torch.mean(region_values)
            max_val = torch.max(region_values)
            min_val = torch.min(region_values)
            std_val = torch.std(region_values)
            region_output = (mean_val + max_val + min_val + std_val) / 4
            outputs.append(region_output)
            
        return torch.stack(outputs)

    def learn(self, training_data, epochs, validation_data=None, real_time_monitor=False):
        batch_size = min(64, len(training_data))  # Batch size maior
        training_data = torch.tensor(training_data, dtype=torch.float, device=device)
        num_samples = training_data.shape[0]
        num_batches = max(1, num_samples // batch_size)
        
        optimizer = torch.optim.AdamW([
            {'params': [cell.position for cell in self.cells],
             'lr': self.learning_rate,
             'weight_decay': self.weight_decay,
             'betas': (0.95, 0.999),  # Momentum maior
             'eps': 1e-8}
        ])
        
        # Scheduler mais agressivo 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,  # Redução mais agressiva
            patience=5,   # Menos paciência
            verbose=True,
            min_lr=self.min_lr,
            threshold=1e-4  # Threshold menor
        )
        
        # Adiciona warm-up
        warmup_epochs = 5
        warmup_factor = 0.3

        best_weights = None
        patience_counter = 0
        l2_lambda = 0.001  # Reduzindo regularização L2
        loss_fn = torch.nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            # Warm-up
            if epoch < warmup_epochs:
                lr = self.learning_rate * (1 + epoch * (1 - warmup_factor) / warmup_epochs)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                    
            epoch_loss = 0
            idx = torch.randperm(num_samples, device=device)
            
            for start in range(0, num_samples, batch_size):
                optimizer.zero_grad()
                end = min(start + batch_size, num_samples)
                batch = training_data[idx[start:end]]
                
                # Normalização do batch
                if start == 0:
                    batch = F.normalize(batch, dim=1)
                
                outputs = self.process_input_batch(batch).clone().requires_grad_(True)
                target_indices = torch.argmax(batch[:, :self.output_dimensions], dim=1)
                cross_loss = loss_fn(outputs, target_indices)
                l2_reg = l2_lambda * sum(torch.sum(cell.position ** 2) for cell in self.cells)
                loss = cross_loss + l2_reg
                loss.backward(retain_graph=True)
                
                # Clipping mais forte
                torch.nn.utils.clip_grad_norm_([cell.position for cell in self.cells], max_norm=0.5)
                
                optimizer.step()
                epoch_loss += loss.item()
                
            if epoch % 2 == 0:
                with torch.no_grad():
                    patterns = self._detect_local_patterns()
                    self._adapt_structure({'local': patterns})
                    self._optimize_cell_positions(patterns)
                    self._update_all_connections()
            
            avg_loss = epoch_loss / num_batches
            self.training_errors.append(avg_loss)
            
            scheduler.step(avg_loss)
            
            if real_time_monitor:
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
            
            if avg_loss < self.best_error - self.min_delta:
                self.best_error = avg_loss
                patience_counter = 0
                best_weights = [cell.position.clone() for cell in self.cells]
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                print(f"Early stopping na época {epoch}")
                if best_weights:
                    for cell, pos in zip(self.cells, best_weights):
                        cell.position = pos.clone()
                break

    def _rotate_structure(self, angle, axes):
        rotation_matrix = self._generate_rotation_matrix(angle, axes)
        positions = torch.stack([cell.position for cell in self.cells])
        rotated = torch.mm(positions, rotation_matrix.t())
        for i, cell in enumerate(self.cells):
            cell.position = rotated[i]

    def _generate_rotation_matrix(self, angle, axes):
        n = self.dimensions
        rotation_matrix = torch.eye(n, device=device)
        i, j = axes
        rotation_matrix[i, i] = torch.cos(angle)
        rotation_matrix[i, j] = -torch.sin(angle)
        rotation_matrix[j, i] = torch.sin(angle)
        rotation_matrix[j, j] = torch.cos(angle)
        return rotation_matrix

    def _morph_structure(self, morph_factor):
        if self.shape_type == GeometricShape.ADAPTIVE:
            morph_factor *= self.morph_strength
            all_positions = torch.stack([cell.position for cell in self.cells], dim=0)
            centroid = torch.mean(all_positions, dim=0)
            for cell in self.cells:
                distance = torch.norm(cell.position - centroid)
                if cell.connections:
                    conn_positions = torch.stack([c.position for c in cell.connections], dim=0)
                    connection_influence = torch.mean(conn_positions, dim=0)
                else:  
                    connection_influence = cell.position
                morph_direction = (connection_influence - cell.position) * morph_factor
                
                # Aumenta força morfológica 
                morph_strength = torch.exp(-distance) * 3.0  # Aumentado
                cell.position += morph_direction * morph_strength
                
                # Reduz ruído
                noise = torch.randn_like(cell.position) * 0.001
                cell.position += noise
                
                self._update_cell_connections(cell)

    def _update_cell_connections(self, cell):
        max_connections = 2 ** self.dimensions
        all_positions = torch.stack([c.position for c in self.cells], dim=0)
        dist = torch.norm(all_positions - cell.position, dim=1)
        sorted_indices = torch.argsort(dist)
        for i in sorted_indices:
            if len(cell.connections) >= max_connections:
                break
            other_cell = self.cells[i]
            if other_cell != cell and other_cell not in cell.connections:
                cell.add_connection(other_cell)
                
    def _detect_local_patterns(self):
        patterns = []
        all_positions = torch.stack([c.position for c in self.cells], dim=0)
        values = torch.stack([c.value for c in self.cells], dim=0)
        for cell in self.cells:
            neighbor_ids = [self.cells.index(n) for n in cell.connections]
        values = torch.stack([c.value for c in self.cells], dim=0)
        for i, cell in enumerate(self.cells):
            neighbor_ids = [self.cells.index(n) for n in cell.connections]
            if neighbor_ids:
                neighbor_vals = values[neighbor_ids]
                neighbor_pos = all_positions[neighbor_ids]
            else:
                neighbor_vals = torch.zeros(1, device=device)
                neighbor_pos = torch.zeros((1, self.dimensions), device=device)

            mean_n = float(torch.mean(neighbor_vals).item()) if len(neighbor_vals) > 0 else 0.0
            std_n = float(torch.std(neighbor_vals).item()) if len(neighbor_vals) > 0 else 0.0

            pattern = {
                'center': cell.position,
                'value': cell.value,
                'mean': mean_n,
                'std': std_n,
                'neighbors': neighbor_pos,
                'symmetry': self._calculate_local_symmetry(neighbor_pos)
            }
            patterns.append(pattern)
        return patterns

    def _calculate_local_symmetry(self, neighbor_positions):
        if not len(neighbor_positions):
            return 0.0
        center = torch.mean(neighbor_positions, dim=0)
        symmetry_score = 0.0
        for pos in neighbor_positions:
            mirror_point = 2 * center - pos
            dist_array = torch.norm(neighbor_positions - mirror_point, dim=1)
            min_dist = torch.min(dist_array)
            symmetry_score += 1.0 / (1.0 + float(min_dist.item()))
        return float(symmetry_score / len(neighbor_positions))

    def _adapt_structure(self, patterns):
        stress = self._calculate_structural_stress()
        if not patterns.get('local'):
            patterns['local'] = self._detect_local_patterns()
        
        if stress > self.stress_threshold:
            self._reorganize_connections()
            
        for cell, pattern in zip(self.cells, patterns['local']):
            # Inicializa força
            force = torch.zeros(self.hidden_dim, device=device)
            
            # Ajusta centro do padrão
            pattern_center = pattern['center'][:self.hidden_dim]
            
            if pattern['symmetry'] < 0.8:
                force += (pattern_center - cell.position) * 0.1
                
            # Calcula média do valor da célula
            cell_value_mean = torch.mean(cell.value)
            
            if abs(cell_value_mean.item() - pattern['mean']) > 0.2:
                direction = torch.sign(torch.tensor(pattern['mean'], device=device) - cell_value_mean)
                force += direction * torch.ones_like(cell.position) * 0.1
                
            # Aplica forças
            cell.position += force * self.learning_rate
            mean_tensor = torch.tensor(pattern['mean'], device=device)
            cell.value = 0.8 * cell.value + 0.2 * mean_tensor

    def _reorganize_connections(self):
        for cell in self.cells:
            self._update_cell_connections(cell)

    def _calculate_structural_stress(self):
        stress = 0.0
        for cell in self.cells:
            for connection in cell.connections:
                dist = torch.norm(cell.position - connection.position)
                stress += float(dist.item())
        return stress

    def _calculate_forces(self, cell, patterns):
        # Inicializa forças com dimensão correta
        forces = torch.zeros(self.hidden_dim, device=device)
        
        for pattern in patterns:
            if 'center' in pattern and 'neighbors' in pattern:
                # Ajusta dimensões do centro e vizinhos
                pattern_center = pattern['center'][:self.hidden_dim]
                pattern_neighbors = pattern['neighbors'][:, :self.hidden_dim]
                
                # Calcula forças
                forces += (pattern_center - cell.position) * 0.1
                forces += torch.sum(pattern_neighbors - cell.position, dim=0) * 0.05
                
        return forces

    def _optimize_cell_positions(self, patterns):
        for cell in self.cells:
            forces = self._calculate_forces(cell, patterns)
            cell.position += forces * self.learning_rate
            self._apply_geometric_constraints(cell)

    def _apply_geometric_constraints(self, cell):
        cell.position = torch.clamp(cell.position, 0.0, 1.0)