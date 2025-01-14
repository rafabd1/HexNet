import numpy as np
from core_network import ComplexGeometricNetwork, GeometricShape
from converters import numeric_converter
from datasets import load_numeric_data, generate_shapes, load_vision_data
import matplotlib.pyplot as plt


def main_numeric():
    data = load_numeric_data()
    X, scaler = numeric_converter(data)
    
    net = ComplexGeometricNetwork(dimensions=X.shape[1])
    net.learn(training_data=X, epochs=50, real_time_monitor=True)
    
    sample = np.array([[25.0, 1014, 67]])
    sample_norm = scaler.transform(sample)
    prediction = net.process_input(sample_norm[0])
    print("Predição (numérica):", scaler.inverse_transform([prediction]))

def main_vision():
    print("\n=== Teste com Visão Computacional ===")
    
    # Carrega dados
    X, y = load_vision_data()
    print(f"Dataset shape: {X.shape}")
    
    # Treina rede
    net = ComplexGeometricNetwork(
        dimensions=X.shape[1],
        shape_type=GeometricShape.ADAPTIVE
    )
    
    net.learn(training_data=X, epochs=100, real_time_monitor=True)
    
    # Testa com novas formas
    test_X, test_y = generate_shapes(n_samples=5)
    
    # Visualiza resultados
    plt.figure(figsize=(15,3))
    for i in range(5):
        plt.subplot(1,5,i+1)
        plt.imshow(test_X[i].reshape(28,28))
        pred = net.process_input(test_X[i])
        plt.title(f'Pred: {np.argmax(pred)}')
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    print("=== Teste com dados numéricos ===")
    main_numeric()
    print("\n=== Teste com dados de visão computacional ===")
    main_vision()