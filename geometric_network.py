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
from network_visualizer import NetworkVisualizer

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
        self.visualizer = NetworkVisualizer()
        
        # Cache e estado
        self.n_cells = None
        self.hidden_dim = None
        self._cell_positions_cache = None
        self._similarity_cache = None
        self.initialized = False
        
        # Hiperparâmetros 
        self.cells = []
        self.topology = None
        self.training_errors = []
        self.validation_errors = []
        self.best_error = float('inf')
        self.stress_threshold = 0.3
        self.learning_rate = 0.01 
        self.min_lr = 0.001 
        self.scheduler_patience = 10
        self.scheduler_factor = 0.7  
        self.weight_decay = 1e-6  
        self.skip_connections = True
        self.skip_alpha = 0.3  
        self.activation = lambda x: torch.relu(x)
        self.patience = 25
        self.min_delta = 0.00001
        self.morph_strength = 2.0 

    def _initialize_topology(self):
        """Inicializa topologia da rede"""
        self._create_hypercube_topology()

    def _create_hypercube_topology(self):
        if self.shape_type == GeometricShape.HYPERCUBE:
            # Definir dimensão oculta primeiro
            self.hidden_dim = 3
            
            # Criar vértices do hipercubo 3D
            base_vertices = list(itertools.product([0, 1], repeat=self.hidden_dim))
            vertices = torch.tensor(base_vertices, dtype=torch.float, device=device)
    
            self.cells = [GeometricCell(self.hidden_dim) for _ in range(len(vertices))]
            
            for i, cell in enumerate(self.cells):
                cell.position = vertices[i].clone()
                # Conectar vértices que diferem em uma coordenada
                for j, other_cell in enumerate(self.cells):
                    diff = torch.sum(torch.abs(vertices[i] - vertices[j]))
                    if diff.item() == 1.0:
                        cell.add_connection(other_cell)
                        
            print(f"Created {len(self.cells)} cells with {self.hidden_dim} dimensions")
            self.topology = {'vertices': vertices, 'cells': self.cells}
        else:
            std = torch.sqrt(torch.tensor(2.0 / (self.input_dimensions + self.hidden_dim), device=device))
            self.cells = []
            
            # Cria células com posições diversas
            positions = []
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
            
            self._update_all_connections()
            
            print(f"Created {len(self.cells)} cells with {self.hidden_dim} dimensions")
            self.topology = {
                'vertices': torch.stack(positions),
                'cells': self.cells
            }
    
    def _analyze_data_complexity(self, data):
        """Analisa complexidade dos dados para definir arquitetura"""
        # Calcula variância/entropia
        variance = torch.var(data, dim=0)
        complexity_score = torch.mean(variance)
        
        # Analisa correlações
        corr_matrix = torch.corrcoef(data.T)
        independent_features = torch.sum(torch.abs(corr_matrix) > 0.5)
        
        # Define número de células baseado na complexidade
        base_cells = max(16, min(256, int(np.sqrt(data.shape[0]))))
        self.n_cells = int(base_cells * (1 + complexity_score.item()))
        
        # Define dimensão oculta
        self.hidden_dim = max(
            32,
            min(256, 
                int(self.input_dimensions * (1 + independent_features.item()/data.shape[1]))
            )
        )

        # Ajustes para séries temporais
        if self.shape_type == GeometricShape.ADAPTIVE:
            self.n_cells = int(self.n_cells * 1.5)
            self.hidden_dim = max(64, self.hidden_dim)
        
        print(f"Adaptative Architecture: {self.n_cells} cells, {self.hidden_dim} hidden dim")
        self.initialized = True
    
    def _update_all_connections(self):
        """Versão vetorizada de update_connections"""
        # Usa cache de posições
        if self._cell_positions_cache is None:
            self._cell_positions_cache = torch.stack([c.position for c in self.cells])
            
        positions = self._cell_positions_cache
        
        # Calcula similaridade coseno vetorizada
        norm = torch.norm(positions, dim=1, keepdim=True)
        normalized = positions / (norm + 1e-8)
        similarity = torch.mm(normalized, normalized.t())
        
        # Atualiza cache
        self._similarity_cache = similarity
        
        # Conexões vetorizadas
        k = min(4, len(self.cells)-1)
        _, indices = torch.topk(similarity, k=k+1, dim=1)
        
        # Limpa e atualiza conexões
        for i, cell in enumerate(self.cells):
            cell.connections = [self.cells[j] for j in indices[i, 1:]]

    def _map_input_to_cells(self, batch_data):
        """Versão otimizada do mapeamento"""
        # Ajuste de dimensões
        if batch_data.shape[1] < self.dimensions:
            pad_cols = self.dimensions - batch_data.shape[1]
            batch_data = F.pad(batch_data, (0, pad_cols), 'constant', 0)
        elif batch_data.shape[1] > self.dimensions:
            batch_data = batch_data[:, :self.dimensions]

        # Se não existir camada de projeção, instancie
        if not hasattr(self, 'projection'):
            self.projection = torch.nn.Linear(self.dimensions, self.hidden_dim).to(device)

        # Usa cache quando possível
        if self._cell_positions_cache is None:
            self._cell_positions_cache = torch.stack([c.position for c in self.cells])
            
        # Vetoriza projeção e normalização
        features = self.projection(batch_data)
        features = F.layer_norm(features, features.shape[1:])
        
        # Ajusta dimensões se necessário
        if features.shape[1] != self._cell_positions_cache.shape[1]:
            pad_size = abs(features.shape[1] - self._cell_positions_cache.shape[1])
            if features.shape[1] < self._cell_positions_cache.shape[1]:
                features = F.pad(features, (0, pad_size), 'constant', 0)
            else:
                features = features[:, :self._cell_positions_cache.shape[1]]
        
        # Calcula atenção vetorizada
        similarity = torch.matmul(features, self._cell_positions_cache.t())
        attention = torch.softmax(
            similarity / torch.sqrt(torch.tensor(self.hidden_dim, device=device)) * 2.0, 
            dim=1
        )
        
        # Update células vetorizado
        weighted_inputs = features.unsqueeze(1) * attention.unsqueeze(2)
        cell_updates = torch.sum(weighted_inputs, dim=0)
        
        # Atualiza todas células de uma vez
        for i, cell in enumerate(self.cells):
            cell.update_value(cell_updates[i])

    def process_input_batch(self, batch_data):
        batch_size = batch_data.shape[0]
        all_preds = []
    
        for i in range(batch_size):
            single_input = batch_data[i].clone().detach().to(device=device, dtype=torch.float).unsqueeze(0)
            self._map_input_to_cells(single_input)
            self._morph_structure(morph_factor=0.1)
    
            values = torch.stack([c.value for c in self.cells], dim=0)
            att = torch.mean(values)
            values = values * att
            feature_map = self._reduce_dimensions(values).unsqueeze(0)
    
            if self.skip_connections:
                input_features = single_input
                
                if not hasattr(self, 'skip_projection'):
                    input_size = input_features.shape[1]  # Número de features
                    hidden_size = input_size * 2
                    
                    self.skip_projection = torch.nn.Sequential(
                        torch.nn.Linear(input_size, hidden_size),
                        torch.nn.ReLU(),
                        torch.nn.Linear(hidden_size, self.output_dimensions)
                    ).to(device)
                
                skip_features = self.skip_projection(input_features)
                feature_map = feature_map.view(-1, self.output_dimensions)
                skip_features = skip_features.view(-1, self.output_dimensions)
                
                # Skip connection mais forte
                gate = torch.sigmoid(torch.sum(feature_map * skip_features, dim=-1, keepdim=True)) * self.skip_alpha
                feature_map = feature_map * (1 - gate) + skip_features * (1 + gate)

            # Adiciona ruído para aumentar variância
            noise = torch.randn_like(feature_map) * 0.01
            feature_map = feature_map + noise
    
            all_preds.append(feature_map)
    
        return torch.cat(all_preds, dim=0)
    
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
            force = torch.zeros(self.hidden_dim, device=device)
            pattern_center = pattern['center'][:self.hidden_dim]
            
            if pattern['symmetry'] < 0.8:
                pattern_center = pattern_center.to(device)
                force += (pattern_center - cell.position) * 0.1
                
            cell.position += force * self.learning_rate
            cell.value = 0.8 * cell.value + 0.2 * torch.tensor(pattern['mean'], device=device)

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

    def learn(self, training_data, epochs, validation_data=None, real_time_monitor=True):
        # Inicializa arquitetura adaptativa
        if not self.initialized:
            self._analyze_data_complexity(training_data)
            self._initialize_topology()

        batch_size = min(64, len(training_data))  # Batch size maior
        training_data = training_data.clone().detach().to(dtype=torch.float, device=device)
        num_samples = training_data.shape[0]
        num_batches = max(1, num_samples // batch_size)
        
        optimizer = torch.optim.AdamW([
            {'params': [cell.position for cell in self.cells],
                'lr': self.learning_rate,
                'weight_decay': self.weight_decay,
                'betas': (0.95, 0.999),
                'eps': 1e-8}
        ])
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=self.scheduler_patience,
            min_lr=self.min_lr
        )
        
        warmup_epochs = 5
        warmup_factor = 0.3

        best_weights = None
        patience_counter = 0
        l2_lambda = 0.001
        mse_loss = torch.nn.MSELoss()
        
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
                targets = batch[:, :self.output_dimensions]
                loss = mse_loss(outputs, targets)
                l2_reg = l2_lambda * sum(torch.sum(cell.position ** 2) for cell in self.cells)
                loss = loss + l2_reg
                loss.backward(retain_graph=True)
                
                torch.nn.utils.clip_grad_norm_([cell.position for cell in self.cells], max_norm=0.5)
                
                optimizer.step()
                epoch_loss += loss.item()
                
            if epoch % 2 == 0:
                with torch.no_grad():
                    patterns = self._detect_local_patterns()
                    self._adapt_structure({'local': patterns})
                    self._optimize_cell_positions(patterns)
                    self._update_all_connections()
                    if real_time_monitor:
                        self.visualizer.update(self)
            
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
                