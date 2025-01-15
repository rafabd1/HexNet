import numpy as np
from geometric_network import ComplexGeometricNetwork, GeometricShape
from converters import numeric_converter
from datasets import load_numeric_data, load_financial_data
import torch
from utils import device
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt


def main_numeric():
    # Carrega e normaliza dados
    data = load_numeric_data()
    X, scaler = numeric_converter(data)
    X_tensor = torch.tensor(X, dtype=torch.float, device=device)
    
    # Cria rede
    net = ComplexGeometricNetwork(
        input_dimensions=X.shape[1],
        output_dimensions=3,
        shape_type=GeometricShape.ADAPTIVE
    )     
    # Agora sim, treina a rede
    net.learn(
        training_data=X_tensor,
        epochs=50,
        real_time_monitor=True
    )
    
    # Predição
    sample = np.array([[25.0, 1014, 67]])
    sample_norm = scaler.transform(sample)
    sample_tensor = torch.tensor(sample_norm, dtype=torch.float, device=device)
    prediction = net.process_input_batch(sample_tensor)
    print("Predição (numérica):", scaler.inverse_transform(prediction.cpu().detach().numpy()))

def main_financial():
    # Carrega dados com split
    (X_train, y_train), (X_test, y_test) = load_financial_data()
    
    # Normaliza
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_norm = scaler_x.fit_transform(X_train)
    X_test_norm = scaler_x.transform(X_test)
    
    y_train_norm = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test_norm = scaler_y.transform(y_test.reshape(-1, 1))
    
    # Modifica preparação dos dados
    X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_norm, dtype=torch.float32)
    
    # Remover concatenação com target
    train_data = X_train_tensor  # Usar apenas features
    
    net = ComplexGeometricNetwork(
        input_dimensions=X_train.shape[1],
        output_dimensions=1,
        shape_type=GeometricShape.ADAPTIVE
    )
    
    net.learn(
        training_data=train_data,  # Apenas features
        epochs=100,
        real_time_monitor=True
    )
    
    # Teste sem concatenação
    X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32)
    predictions = net.process_input_batch(X_test_tensor)
    predictions = scaler_y.inverse_transform(predictions.cpu().detach().numpy())
    
    # Métricas
    mse = mean_squared_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    print(f"\nMétricas de Teste:")
    print(f"MSE: {mse:.2f}")
    print(f"MAPE: {mape*100:.2f}%")
    
    # Visualização
    plt.figure(figsize=(12,6))
    plt.plot(y_test, label='Real')
    plt.plot(predictions, label='Previsto')
    plt.title('S&P500 - Previsão vs Real')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # print("=== Teste com dados numéricos ===")
    # main_numeric()
    print("\n=== Teste com dados financeiros ===")
    main_financial()