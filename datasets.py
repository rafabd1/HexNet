import numpy as np
import yfinance as yf
import pandas as pd

def load_financial_data(test_size=0.2):
    """Carrega dados do índice S&P500 - último ano"""
    # Download 1 ano de dados
    sp500 = yf.download('^GSPC', 
                        start='2023-01-01', 
                        end='2024-01-01',
                        interval='1d')
    
    # Features técnicas mais importantes
    sp500['Returns'] = sp500['Close'].pct_change()
    sp500['MA20'] = sp500['Close'].rolling(window=20).mean()
    sp500['RSI'] = calculate_rsi(sp500['Close'])
    sp500['Volatility'] = sp500['Returns'].rolling(window=20).std()
    
    sp500 = sp500.dropna()
    
    # Features e target
    X = sp500[['Returns', 'MA20', 'RSI', 'Volatility']].values
    y = sp500['Close'].values
    
    # Split treino/teste
    split_idx = int(len(X) * (1-test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return (X_train, y_train), (X_test, y_test)

def calculate_rsi(data, periods=14):
    """Calcula RSI"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def load_numeric_data():
    # Exemplo de dados climáticos
    data = [
        [25.5, 1013, 65],
        [24.8, 1015, 68],
        [26.2, 1010, 70],
        [25.1, 1012, 67]
    ]
    return np.array(data)
