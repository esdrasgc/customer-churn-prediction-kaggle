# Implementação do MLP

Este projeto utiliza `MLPClassifier` do scikit-learn com otimização via Optuna, conforme implementado em `notebooks/3_train_egc.ipynb` e `notebooks/train_mlp_cv_optuna.py`.

## Implementação Principal

### Modelo Base
- **Biblioteca**: `sklearn.neural_network.MLPClassifier`
- **Código principal**: `notebooks/train_mlp_cv_optuna.py`
- **Hiperparâmetros configuráveis**:
  - `hidden_layer_sizes`: arquitetura das camadas ocultas
  - `activation`: {`relu`, `tanh`, `logistic` (sigmoid)}
  - `solver`: {`adam`, `lbfgs`, `sgd`}
  - `alpha`: regularização L2
  - `learning_rate_init`: taxa de aprendizado inicial
  - `random_state`: seed para reprodutibilidade

### Pré-processamento Integrado
Conforme `_build_preprocessor()` em `train_mlp_cv_optuna.py`:
- **Numéricos**: `SimpleImputer(median)` + `StandardScaler`
- **Categóricos**: `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown='ignore')`

## Otimização de Hiperparâmetros com Optuna

Conforme `notebooks/3_train_egc.ipynb`, utilizamos **Optuna** para busca de hiperparâmetros:

### Espaço de Busca
```python
'n_layers': [1, 5]  # número de camadas ocultas
'layer_i_size': [50, 500]  # neurônios por camada
'activation': ['relu', 'tanh', 'logistic']
'solver': ['adam', 'lbfgs']
'alpha': [1e-6, 1e-1]  # log scale
'learning_rate_init': [1e-5, 1e-1]  # log scale
```

### Melhores Hiperparâmetros Encontrados

Após **10 trials** com otimização minimizando **validation log loss**:

```python
{
    'n_layers': 3,
    'layer_0_size': 411,
    'layer_1_size': 359,
    'layer_2_size': 361,
    'activation': 'logistic',
    'solver': 'lbfgs',
    'alpha': 7.78472202696525e-05,
    'learning_rate_init': 3.2734483532096865e-05
}
```

- **Arquitetura final**: 3 camadas ocultas com [411, 359, 361] neurônios
- **Função de ativação**: `logistic` (sigmoid)
- **Solver**: `lbfgs` (otimizador de segunda ordem)
- **Melhor val log loss**: 0.693145

## Estratégia de Treinamento

### Split Balanceado
```python
train shape: (3510, 20) class counts: Counter({1: 1755, 0: 1755})
val   shape: (750, 20)  class counts: Counter({0: 375, 1: 375})
test  shape: (750, 20)  class counts: Counter({0: 375, 1: 375})
```

### Pipeline Completo
1. Split estratificado 70/15/15 com balanceamento perfeito por classe
2. Cross-validation (4 folds) no conjunto de treino para robustez
3. Treinamento no conjunto completo de treino
4. Avaliação em validação e teste
5. Re-treino final em treino+validação para submissão

## Treinamento por Época (Alternativa)

Em `train_mlp_cv_optuna.py`, implementamos também `train_mlp_with_metrics()` que:

- Usa `partial_fit` para treinamento iterativo por mini-batches
- Coleta métricas (loss, accuracy, F1) a cada época
- Implementa early stopping monitorando `val_f1` com `patience`
- Restaura melhores pesos ao final
