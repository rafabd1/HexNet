import numpy as np
from skimage.draw import polygon

def load_numeric_data():
    # Exemplo de dados climáticos
    data = [
        [25.5, 1013, 65],
        [24.8, 1015, 68],
        [26.2, 1010, 70],
        [25.1, 1012, 67]
    ]
    return np.array(data)


def generate_shapes(size=28, n_samples=20):
    """Gera formas geométricas sintéticas"""
    shapes = []
    labels = []
    
    for _ in range(n_samples):
        # Cria imagem vazia
        img = np.zeros((size, size))
        
        # Escolhe forma aleatória
        shape_type = np.random.choice(['circle', 'square', 'triangle'])
        
        # Gera parâmetros aleatórios
        center = np.random.randint(size//4, 3*size//4, 2)
        radius = np.random.randint(3, size//4)
        
        if shape_type == 'circle':
            # Desenha círculo
            y, x = np.ogrid[:size, :size]
            dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            img[dist <= radius] = 1
            labels.append(0)
            
        elif shape_type == 'square':
            # Desenha quadrado
            x1, y1 = center[0] - radius, center[1] - radius
            x2, y2 = center[0] + radius, center[1] + radius
            img[y1:y2, x1:x2] = 1
            labels.append(1)
            
        else:
            # Desenha triângulo
            pts = np.array([
                [center[0], center[1] - radius],
                [center[0] - radius, center[1] + radius],
                [center[0] + radius, center[1] + radius]
            ])
            rr, cc = polygon(pts[:,0], pts[:,1], img.shape)
            img[rr, cc] = 1
            labels.append(2)
            
        shapes.append(img.flatten())
    
    return np.array(shapes), np.array(labels)

def load_vision_data():
    """Carrega dataset de visão computacional"""
    X, y = generate_shapes()
    return X, y