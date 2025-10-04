# Curvas de Erro e Visualização

Esta seção apresenta as curvas de aprendizado e análise de convergência do modelo.

## Curvas de Treinamento

### Função para Gerar Curvas

Implementado em `train_mlp_cv_optuna.py`:

```python
def plot_history(history: List[EpochMetrics]):
    epochs = [m.epoch for m in history]
    tr_loss = [m.train_loss for m in history]
    va_loss = [m.val_loss for m in history]
    tr_acc  = [m.train_acc  for m in history]
    va_acc  = [m.val_acc    for m in history]
    
    # Loss curves
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, tr_loss, label="Train loss")
    plt.plot(epochs, va_loss, label="Val loss")
    plt.xlabel("Épocas")
    plt.ylabel("Log-loss")
    plt.title("Curva de perda por época")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Accuracy curves
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, tr_acc, label="Train acc")
    plt.plot(epochs, va_acc, label="Val acc")
    plt.xlabel("Épocas")
    plt.ylabel("Acurácia")
    plt.title("Acurácia por época")
    plt.legend()
    plt.grid(True)
    plt.show()
```

## Observações sobre Convergência

### Modelo Final (LBFGS)

O solver **LBFGS** utilizado no modelo final:

- Convergência rápida (geralmente < 100 iterações)
- Não requer tuning manual de learning rate
- Otimizador de segunda ordem (usa informação da Hessiana)
- Ideal para datasets de tamanho médio como o nosso (8k amostras)

### Treinamento por Época (Adam/SGD)

Quando usando `train_mlp_with_metrics()` com Adam:

- Curvas mais suaves e grad uais
- Early stopping previne overtraining
- Monitoramento de `val_f1` garante parada no ponto ótimo

## Análise de Threshold (F1 Optimization)

Para otimizar o F1-score (métrica da competição), pode-se variar o threshold de classificação:

```python
import numpy as np
from sklearn.metrics import f1_score

# Obter probabilidades
y_val_proba = model.predict_proba(X_val)[:, 1]

# Testar diferentes thresholds
thresholds = np.linspace(0.3, 0.7, 41)
f1_scores = []

for thresh in thresholds:
    y_pred = (y_val_proba >= thresh).astype(int)
    f1 = f1_score(y_val, y_pred)
    f1_scores.append(f1)

best_thresh = thresholds[np.argmax(f1_scores)]
print(f"Melhor threshold: {best_thresh:.3f}")
print(f"F1-score máximo: {max(f1_scores):.4f}")
```

### Threshold Padrão

Neste projeto, utilizamos **threshold = 0.5** (padrão), dado que:

- Classes foram balanceadas artificialmente nos splits
- Log loss já calibra bem as probabilidades
- Simplifica reprodução dos resultados

!!! tip "Como exportar os plots"
    Para adicionar visualizações nesta página:
    
    1. Rode `train_mlp_with_metrics()` em um notebook
    2. Chame `plot_history(history)`
    3. Salve as figuras: `plt.savefig('docs/plots_train/loss_curves.png', dpi=150, bbox_inches='tight')`
    4. Adicione aqui via markdown
