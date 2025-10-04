# Treinamento do Modelo

Esta seção descreve o processo de treinamento do MLP e os desafios encontrados.

## Loop de Treinamento

### Abordagem 1: Optuna com LBFGS (Modelo Final)

Conforme `notebooks/3_train_egc.ipynb`:

```python
from train_mlp_cv_optuna import run_training

results = run_training(
    X, y,
    n_trials=10,
    random_state=42,
    cv_folds=4,
    verbose=True
)
```

**Características**:

- Otimização automática via Optuna (10 trials)
- Split balanceado 70/15/15 com classes equilibradas
- Cross-validation 4-fold no conjunto de treino
- Métrica de otimização: **validation log loss** (minimizar)
- Solver final: **LBFGS** (otimizador de segunda ordem, não requer tuning de learning rate)

### Abordagem 2: Treinamento por Época com Early Stopping

Implementado em `train_mlp_with_metrics()` (`train_mlp_cv_optuna.py`):

```python
clf, history, final_stats = train_mlp_with_metrics(
    X_train, y_train, X_val, y_val,
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    learning_rate_init=1e-3,
    batch_size=64,
    max_epochs=200,
    patience=20,
    monitor="val_f1",
    monitor_mode="max"
)
```

**Métricas coletadas por época**:

- `train_loss`, `val_loss`
- `train_acc`, `val_acc`
- `train_f1`, `val_f1`

**Early Stopping**:

- Monitora `val_f1` (ou `val_loss`)
- Parâmetro `patience`: número de épocas sem melhora antes de parar
- Restaura os melhores pesos ao final

## Inicialização e Regularização

- **Inicialização**: `random_state=42` fixo para reprodutibilidade
- **Regularização L2**: `alpha` controla weight decay
  - Valor ótimo encontrado: `7.78e-05`

## Artefatos Salvos

Diretório: `notebooks/artifacts/`

- `best_model_train.pkl`: modelo treinado apenas no conjunto de treino
- `final_model_trainval.pkl`: modelo retreinado em treino+validação para submissão final
- `optuna_study.pkl`: histórico completo da otimização Optuna

## Desafios e Observações

### 1. Convergência do Modelo
- **LBFGS** mostrou convergência rápida e estável
- Solvers como **Adam** requerem tuning cuidadoso de `learning_rate_init`
- Modelos muito profundos (4-5 camadas) apresentaram maior dificuldade de convergência

### 2. Overfitting
- Dataset relativamente pequeno (8k amostras) exige regularização adequada
- `alpha` (L2) foi crucial para evitar overfitting
- Early stopping baseado em `val_f1` preveniu treinamento excessivo

### 3. Balanceamento de Classes
- Split balanceado 50/50 em cada partição melhorou estabilidade das métricas
- Evita viés do modelo para a classe majoritária

## Reprodutibilidade

- `random_state=42` usado consistentemente em:
  - Splits de dados
  - Inicialização do modelo
  - Cross-validation
- Permite reprodução exata dos resultados
