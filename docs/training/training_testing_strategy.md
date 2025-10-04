# Estratégia de Treinamento e Teste

Esta seção detalha como os dados foram divididos e como garantimos avaliação robusta do modelo.

## Divisão dos Dados

### Estratégia de Split

Utilizamos **split balanceado 70/15/15** implementado em `stratified_custom_split_indices()` (`train_mlp_cv_optuna.py`):

```python
train_idx, val_idx, test_idx = stratified_custom_split_indices(
    y, 
    random_state=42,
    train_frac=0.70,
    val_frac=0.15,
    test_frac=0.15
)
```

**Características**:

- Cada partição (treino/validação/teste) contém **exatamente o mesmo número de positivos e negativos**
- Baseado na classe minoritária para garantir balanceamento perfeito
- Garante `random_state=42` para reprodutibilidade

### Tamanhos dos Conjuntos

Conforme `notebooks/3_train_egc.ipynb`:

```
train shape: (3510, 20) class counts: {1: 1755, 0: 1755}
val   shape: (750, 20)  class counts: {0: 375, 1: 375}
test  shape: (750, 20)  class counts: {0: 375, 1: 375}
```

- **Treino**: 3.510 amostras (70% balanceado)
- **Validação**: 750 amostras (15% balanceado)
- **Teste**: 750 amostras (15% balanceado)

## Cross-Validation

### Validação Cruzada no Treino

Durante a otimização Optuna, aplicamos `StratifiedKFold` (4 folds) **apenas no conjunto de treino**:

```python
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    pipe, X_train, y_train, 
    cv=cv, 
    scoring="accuracy", 
    n_jobs=-1
)
```

**Objetivo**:

- Sinal de robustez durante busca de hiperparâmetros
- Evita overfitting na seleção de modelo
- Não "contamina" validação/teste

## Métricas Consideradas

- **Otimização primária**: `log_loss` (validation)
- **Métricas secundárias**: `accuracy`, `roc_auc`, `average_precision`, `f1`
- **Métrica da competição**: **F1-score**

## Controle de Overfitting

### Técnicas Aplicadas

1. **Regularização L2** (`alpha=7.78e-05`)
2. **Cross-validation** durante busca de hiperparâmetros
3. **Split balanceado** para métricas estáveis
4. **Early stopping** (quando usando treinamento por época):
   - Monitor: `val_f1` ou `val_loss`
   - `patience`: número de épocas sem melhora
   - Restaura melhores pesos

## Pipeline de Avaliação

### Fase 1: Desenvolvimento
1. Treinar no **conjunto de treino** (3.510 amostras)
2. Selecionar hiperparâmetros usando **validação** (750 amostras)
3. Avaliar performance no **teste** (750 amostras)

### Fase 2: Submissão Final
1. Retreinar modelo com melhores hiperparâmetros em **treino + validação** (4.260 amostras)
2. Avaliar no **teste** para confirmar
3. Aplicar no conjunto de teste da competição (2.000 amostras)

## Reprodutibilidade

**Seeds fixas em todos os componentes**:

- `random_state=42` em:
  - `stratified_custom_split_indices()`
  - `MLPClassifier()`
  - `StratifiedKFold()`
  - `Optuna.Study(seed=42)` (quando aplicável)

**Dados salvos**:

- `data/X.pickle`, `data/y.pickle`
- `notebooks/artifacts/*.pkl`
