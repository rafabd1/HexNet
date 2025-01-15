import numpy as np
from geometric_network import ComplexGeometricNetwork, GeometricShape
from converters import numeric_converter
from datasets import load_numeric_data, generate_shapes, load_vision_data
import matplotlib.pyplot as plt
import torch


def main_numeric():
    data = load_numeric_data()
    X, scaler = numeric_converter(data)
    
    net = ComplexGeometricNetwork(
        input_dimensions=X.shape[1],
        output_dimensions=3,
        shape_type=GeometricShape.ADAPTIVE
    )
    net.learn(
        training_data=X,
        epochs=100,
        real_time_monitor=True
    )
    
    sample = np.array([[25.0, 1014, 67]])
    sample_norm = scaler.transform(sample)
    prediction = net.process_input_batch(sample_norm)
    print("Predição (numérica):", scaler.inverse_transform(prediction.cpu().detach().numpy()))

def main_vision():
    # Carrega e prepara dados
    X, y = load_vision_data()
    print(f"Dataset shape: {X.shape}")
    
    # Converte labels para one-hot encoding
    y_onehot = np.zeros((len(y), 3))
    for i, label in enumerate(y):
        y_onehot[i, label] = 1
    
    net = ComplexGeometricNetwork(
        input_dimensions=X.shape[1],
        output_dimensions=3,
        shape_type=GeometricShape.ADAPTIVE
    )
    
    # Treina com dados e labels
    # Treina apenas com features
    X_norm = (X - X.min()) / (X.max() - X.min())
    net.learn(
        training_data=X_norm,
        epochs=100, 
        real_time_monitor=True
    )
    
    # Testa formas
    size = 32
    test_X, test_y = generate_shapes()
    test_X = (test_X - test_X.min()) / (test_X.max() - test_X.min())
    
    shape_names = {0:'Círculo', 1:'Quadrado', 2:'Triângulo'}
    
    plt.figure(figsize=(20,4))
    correct = 0
    
    predictions = []  # Lista para armazenar predições
    
    for i in range(5):
        # Normaliza cada imagem individualmente
        img = test_X[i].reshape(1, -1)
        img = (img - img.min()) / (img.max() - img.min())
        
        # Processa com reset de estado
        with torch.no_grad():
            pred = net.process_input_batch(img)
            pred = pred.squeeze()
            pred = torch.softmax(pred, dim=0)
            pred_idx = int(torch.argmax(pred).item())
            predictions.append(pred_idx)  # Armazena predição
            
        # Verifica se predições são diferentes    
        if i > 0 and predictions[i] == predictions[i-1]:
            print(f"Aviso: Predições iguais para imagens {i-1} e {i}")
            
        plt.subplot(1,5,i+1)
        plt.imshow(test_X[i].reshape(size,size), cmap='gray')
        
        # Avalia
        if pred_idx == test_y[i]:
            correct += 1
            color = 'green'
        else:
            color = 'red'
        
        # Mostra confiança da predição
        conf = float(pred[pred_idx])
        plt.title(
            f'Pred: {shape_names[pred_idx]} ({conf:.1%})\nReal: {shape_names[test_y[i]]}',
            color=color
        )
        plt.axis('off')
    
    acc = correct/5
    plt.suptitle(f'Acurácia: {acc:.0%} ({correct}/5)', fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("=== Teste com dados numéricos ===")
    main_numeric()
    print("\n=== Teste com dados de visão computacional ===")
    main_vision()