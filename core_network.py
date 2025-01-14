import cupy as cp
import itertools
from enum import Enum
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

print(f"CUDA available: {cp.cuda.is_available()}")
print(f"CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
print(f"Current CUDA device: {cp.cuda.runtime.getDevice()}")

cp.random.seed(42)  # Fixa seed para reprodutibilidade

class GeometricShape(Enum):
    HYPERCUBE = "hypercube"
    DODECAHEDRON = "dodecahedron"
    TESSERACT = "tesseract"
    ADAPTIVE = "adaptive"

class GeometricCell:
    def __init__(self, dimensions, value=0.0):
        self.position = cp.zeros(dimensions)
        self.value = value
        self.bias = 0.01
        self.connections = []
        self.gradient = cp.zeros(dimensions)
        self.momentum = 0.8
        
    def update_value(self, new_value, learning_rate=0.01):
        l2_reg = 0.01 * cp.sum(self.position ** 2)
        old_value = self.value
        self.value = new_value * (1 - self.momentum) + old_value * self.momentum - l2_reg
        self.gradient = cp.clip(self.gradient * self.momentum, -0.1, 0.1)
    
    def add_connection(self, other_cell):
        self.connections.append(other_cell)

class ComplexGeometricNetwork:
    def __init__(self, dimensions, shape_type=GeometricShape.HYPERCUBE):
        self.dimensions = dimensions
        self.shape_type = shape_type
        self.cells = []
        self.memory_cells = []
        self.topology = self._initialize_topology()
        self.training_errors = []
        self.validation_errors = []
        self.best_error = float('inf')
        self.stress_threshold = 0.3
        self.learning_rate = 0.005
        self.activation = lambda x: cp.tanh(x / 2)
        self.momentum = 0.85
        self.patience = 50
        self.min_delta = 0.0005

    def _map_input_to_cells(self, input_data):
        input_data = cp.asarray(input_data).flatten()[:self.dimensions]
        if len(input_data) < self.dimensions:
            input_data = cp.pad(input_data, (0, self.dimensions - len(input_data)), 'constant')
        input_data = input_data / cp.sqrt(self.dimensions)
        centroid = cp.mean(cp.stack([cell.position for cell in self.cells], axis=0), axis=0)
        
        for cell in self.cells:
            distance = cp.linalg.norm(cell.position - centroid)
            weight = 1.0 / (1.0 + distance**2)
            dimensional_weights = cp.exp(-cp.abs(cell.position))
            dimensional_weights = dimensional_weights[:len(input_data)]
            weighted_input = cp.sum(input_data * dimensional_weights) * weight
            cell.update_value(self.activation(weighted_input), self.learning_rate)

    def process_input(self, input_data):
        self._map_input_to_cells(input_data)
        self._morph_structure(0.1)
        
        prediction = cp.zeros(self.dimensions)
        for dim in range(self.dimensions):
            cells_dim = sorted(
                self.cells,
                key=lambda c: cp.abs(c.position[dim] - 0.5),
                reverse=True
            )[:2]
            weights = [1.0/(1.0 + cp.abs(c.position[dim])) for c in cells_dim]
            total_weight = sum(weights)
            prediction[dim] = sum(c.value * w/total_weight for c, w in zip(cells_dim, weights))
        
        return self.activation(prediction)

    def learn(self, training_data, epochs, validation_data=None, real_time_monitor=False):
        best_weights = None
        patience_counter = 0
        min_lr = 0.001
        max_lr = 0.01
        
        training_data = cp.asarray(training_data)
        self.input_dim = training_data.shape[1]
        
        for epoch in range(epochs):
            current_lr = max_lr * cp.exp(-epoch/100)
            current_lr = max(current_lr, min_lr)
            self.learning_rate = float(current_lr)
            
            err_sum = 0.0
            angles = cp.linspace(0, cp.pi/8, 4)
            for angle in angles:
                for i in range(min(self.dimensions, self.input_dim) - 1):
                    self._rotate_structure(angle, (i, i+1))
            
            for sample in training_data:
                sample = sample[:self.dimensions]
                if len(sample) < self.dimensions:
                    sample = cp.pad(sample, (0, self.dimensions - len(sample)), 'constant')
                output = self.process_input(sample)
                error = cp.mean((output[:len(sample)] - sample) ** 2)
                err_sum += error
                
                if error > 0.1 * (1 - epoch/epochs):
                    patterns = self._detect_local_patterns()
                    self._adapt_structure({'local': patterns})
                    self._optimize_cell_positions(patterns)
            
            avg_error = err_sum / len(training_data)
            self.training_errors.append(float(avg_error))
            
            if real_time_monitor:
                print(f"Epoch {epoch}: Error = {float(avg_error):.4f} (lr={float(current_lr):.6f})")
    
            if avg_error < self.best_error - self.min_delta:
                self.best_error = float(avg_error)
                patience_counter = 0
                best_weights = [cell.position.copy() for cell in self.cells]
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                print(f"Early stopping na época {epoch}")
                if best_weights:
                    for cell, pos in zip(self.cells, best_weights):
                        cell.position = pos
                break

    def _initialize_topology(self):
        if self.shape_type == GeometricShape.HYPERCUBE:
            return self._create_hypercube_topology()
        elif self.shape_type == GeometricShape.DODECAHEDRON:
            return self._create_dodecahedron_topology()
        elif self.shape_type == GeometricShape.ADAPTIVE:
            return self._create_hypercube_topology()

    def _create_hypercube_topology(self):
        max_dims = min(self.dimensions, 4)
        print(f"Criando topologia com {max_dims} dimensões...")
        
        vertices = cp.array(list(itertools.product([0, 1], repeat=max_dims)), dtype=float)
        print(f"Vertices criados: {len(vertices)}")
        
        if max_dims < self.dimensions:
            padding = cp.zeros((len(vertices), self.dimensions - max_dims))
            vertices = cp.hstack((vertices, padding))
            
        cells = [GeometricCell(self.dimensions) for _ in range(len(vertices))]
        print(f"Células criadas: {len(cells)}")
        
        for i, cell in enumerate(cells):
            cell.position = vertices[i]
            connections = 0
            for j, other_cell in enumerate(cells):
                if cp.sum(cp.abs(vertices[i] - vertices[j])) == 1:
                    cell.add_connection(other_cell)
                    connections += 1
            
        self.cells = cells
        print(f"Topologia criada: {len(cells)} células")
        return {'vertices': vertices, 'cells': cells}

    def _create_dodecahedron_topology(self):
        return self._create_hypercube_topology()

    def _initialize_transformations(self):
        return {
            'rotate': self._rotate_structure,
            'morph': self._morph_structure,
            'reflect': self._create_hypercube_topology 
        }

    def _generate_rotation_matrix(self, angle, axes):
        n = self.dimensions
        rotation_matrix = cp.eye(n)
        i, j = axes
        rotation_matrix[i, i] = cp.cos(angle)
        rotation_matrix[i, j] = -cp.sin(angle)
        rotation_matrix[j, i] = cp.sin(angle)
        rotation_matrix[j, j] = cp.cos(angle)
        return rotation_matrix

    def _rotate_structure(self, angle, axes):
        rotation_matrix = self._generate_rotation_matrix(angle, axes)
        for cell in self.cells:
            cell.position = cp.dot(rotation_matrix, cell.position)

    def _morph_structure(self, morph_factor):
        if self.shape_type == GeometricShape.ADAPTIVE:
            centroid = cp.mean(cp.stack([cell.position for cell in self.cells], axis=0), axis=0)
            for cell in self.cells:
                distance = cp.linalg.norm(cell.position - centroid)
                if cell.connections:
                    connection_influence = cp.mean(cp.stack([c.position for c in cell.connections], axis=0), axis=0)
                else:
                    connection_influence = cell.position
                morph_direction = (connection_influence - cell.position) * morph_factor
                morph_strength = 1.0 / (1.0 + cp.exp(-distance))
                cell.position += morph_direction * morph_strength
                self._update_cell_connections(cell)

    def _update_cell_connections(self, cell):
        max_connections = 2 ** self.dimensions
        distances = [(c, cp.linalg.norm(cell.position - c.position)) for c in self.cells if c != cell]
        closest = sorted(distances, key=lambda x: float(x[1]))[:max_connections]
        cell.connections = [c for c, _ in closest]

    def _calculate_structural_stress(self):
        stress = 0.0
        for cell in self.cells:
            for connection in cell.connections:
                stress += float(cp.linalg.norm(cell.position - connection.position))
        return stress

    def _detect_local_patterns(self):
        patterns = []
        for cell in self.cells:
            neighbor_values = [c.value for c in cell.connections]
            neighbor_positions = [c.position for c in cell.connections]
            pattern = {
                'center': cell.position,
                'value': cell.value,
                'mean': float(cp.mean(cp.array(neighbor_values))) if neighbor_values else 0.0,
                'std': float(cp.std(cp.array(neighbor_values))) if neighbor_values else 0.0,
                'neighbors': cp.stack(neighbor_positions) if neighbor_positions else cp.zeros((0,)),
                'symmetry': self._calculate_local_symmetry(cell, neighbor_positions)
            }
            patterns.append(pattern)
        return patterns

    def _calculate_local_symmetry(self, _, neighbor_positions):
        if not neighbor_positions:
            return 0.0
        center = cp.mean(cp.stack(neighbor_positions, axis=0), axis=0)
        symmetry_score = 0.0
        for pos in neighbor_positions:
            mirror_point = 2 * center - pos
            min_dist = min(cp.linalg.norm(p - mirror_point) for p in neighbor_positions)
            symmetry_score += 1 / (1 + float(min_dist))
        return symmetry_score / len(neighbor_positions)

    def _adapt_structure(self, patterns):
        stress = self._calculate_structural_stress()
        if not patterns.get('local'):
            patterns['local'] = self._detect_local_patterns()
        if stress > self.stress_threshold:
            self._reorganize_connections()
        for cell, pattern in zip(self.cells, patterns['local']):
            force = cp.zeros(self.dimensions)
            if pattern['symmetry'] < 0.8:
                force += (pattern['center'] - cell.position) * 0.1
            if abs(cell.value - pattern['mean']) > 0.2:
                force += cp.sign(pattern['mean'] - cell.value) * 0.1
            cell.position += force * self.learning_rate
            cell.value = 0.8 * cell.value + 0.2 * pattern['mean']

    def _reorganize_connections(self):
        for cell in self.cells:
            self._update_cell_connections(cell)

    def _calculate_error(self, output, target):
        output = cp.asarray(output)
        target = cp.asarray(target).flatten()[:len(output)]
        return float(cp.mean((output - target) ** 2))

    def _calculate_forces(self, cell, patterns):
        forces = cp.zeros(self.dimensions)
        for pattern in patterns:
            if 'center' in pattern and 'neighbors' in pattern:
                forces += pattern['center'] - cell.position
                forces += cp.sum(pattern['neighbors'] - cell.position, axis=0)
        return forces

    def _optimize_cell_positions(self, patterns):
        for cell in self.cells:
            forces = self._calculate_forces(cell, patterns)
            cell.position += forces * self.learning_rate
            self._apply_geometric_constraints(cell)

    def _apply_geometric_constraints(self, cell):
        cell.position = cp.clip(cell.position, 0, 1)

    def analyze_sensitivity(self, input_data, delta=0.01):
        input_data = cp.asarray(input_data)
        base_output = self.process_input(input_data)
        sensitivities = []
        for i in range(len(input_data.flatten())):
            perturbed_input = cp.array(input_data).flatten()
            perturbed_input[i] += delta
            new_output = self.process_input(perturbed_input.reshape(input_data.shape))
            sensitivity = (new_output - base_output) / delta
            sensitivities.append((i, sensitivity))
        return sensitivities

    def plot_learning_curves(self):
        plt.figure()
        plt.plot(self.training_errors, label='Training Error')
        if len(self.validation_errors) > 0:
            plt.plot(self.validation_errors, label='Validation Error', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.legend()
        plt.show()
