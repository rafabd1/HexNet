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


import numpy as np
from skimage.draw import disk, polygon
from skimage.transform import resize
from scipy.ndimage import gaussian_filter

def generate_shapes(size=32, n_samples=50):
    shapes = []
    labels = []
    
    # Aumenta resolução inicial
    high_res = size * 4  # 4x resolução final
    
    shape_mapping = {
        'circle': 0,
        'square': 1, 
        'triangle': 2
    }
    
    for _ in range(n_samples):
        img = np.zeros((high_res, high_res))
        center = np.array([high_res//2, high_res//2])
        radius = high_res//4  # Proporção mais precisa
        
        shape_type = np.random.choice(list(shape_mapping.keys()))
        
        if shape_type == 'circle':
            # Círculo mais suave
            y, x = np.ogrid[:high_res, :high_res]
            dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            img[dist <= radius] = 1
            # Suaviza bordas
            img = gaussian_filter(img, sigma=1.5)
            
        elif shape_type == 'square':
            # Quadrado com rotação
            angle = np.random.uniform(0, 360)
            points = np.array([
                [-radius, -radius],
                [radius, -radius],
                [radius, radius],
                [-radius, radius]
            ]) * 0.8  # Ajusta tamanho
            
            rot = np.array([
                [np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                [np.sin(np.radians(angle)), np.cos(np.radians(angle))]
            ])
            points = np.dot(points, rot)
            points += center
            rr, cc = polygon(points[:,0], points[:,1], img.shape)
            img[rr, cc] = 1
            img = gaussian_filter(img, sigma=0.8)
            
        else:  # triangle
            # Triângulo equilátero mais preciso
            side = radius * 1.5
            height = side * np.sqrt(3)/2
            points = np.array([
                [center[0], center[1] - height/2],
                [center[0] - side/2, center[1] + height/2],
                [center[0] + side/2, center[1] + height/2]
            ])
            
            angle = np.random.uniform(0, 360)
            rot = np.array([
                [np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                [np.sin(np.radians(angle)), np.cos(np.radians(angle))]
            ])
            points = np.dot(points - center, rot) + center
            rr, cc = polygon(points[:,0], points[:,1], img.shape)
            img[rr, cc] = 1
            img = gaussian_filter(img, sigma=1.0)
        
        # Redimensiona com antialiasing
        img = resize(img, (size, size), anti_aliasing=True, mode='constant')
        img = (img > 0.5).astype(float)  # Binariza novamente
        
        # Adiciona ruído aleatório para garantir imagens diferentes
        noise = np.random.normal(0, 0.01, (size, size))
        img = img + noise
        img = np.clip(img, 0, 1)
        
        shapes.append(img.flatten())
        labels.append(shape_mapping[shape_type])
    
    return np.array(shapes), np.array(labels)

def load_vision_data():
    """Carrega dataset de visão computacional"""
    X, y = generate_shapes()
    return X, y