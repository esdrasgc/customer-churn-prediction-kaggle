# Métricas de Avaliação

Esta seção apresenta as métricas de desempenho do modelo no conjunto de teste.

## Métricas no Conjunto de Teste

Conforme `notebooks/3_train_egc.ipynb`, avaliamos o modelo em três conjuntos:

### Modelo Treinado Apenas em TRAIN

**Train Set** (3.510 amostras):
```
accuracy = 0.4929
logloss  = 0.693151

              precision    recall  f1-score   support

           0     0.4927    0.4826    0.4876      1755
           1     0.4930    0.5031    0.4980      1755

    accuracy                         0.4929      3510
   macro avg     0.4929    0.4929    0.4928      3510
weighted avg     0.4929    0.4929    0.4928      3510
```

**Validation Set** (750 amostras):
```
accuracy = 0.5267
logloss  = 0.693145

              precision    recall  f1-score   support

           0     0.5278    0.5067    0.5170       375
           1     0.5256    0.5467    0.5359       375

    accuracy                         0.5267       750
   macro avg     0.5267    0.5267    0.5265       750
weighted avg     0.5267    0.5267    0.5265       750
```

**Test Set** (750 amostras):
```
accuracy = 0.5067
logloss  = 0.693167

              precision    recall  f1-score   support

           0     0.5061    0.4987    0.5024       375
           1     0.5073    0.5147    0.5109       375

    accuracy                         0.5067       750
   macro avg     0.5067    0.5067    0.5066       750
weighted avg     0.5067    0.5067    0.5066       750
```

### Modelo Final (Retreinado em TRAIN+VAL)

Após retreinar em treino+validação (4.260 amostras):

**Test Set** (750 amostras):
```
accuracy = 0.5040
logloss  = 0.693166

              precision    recall  f1-score   support

           0     0.5032    0.4960    0.4996       375
           1     0.5048    0.5120    0.5084       375

    accuracy                         0.5040       750
   macro avg     0.5040    0.5040    0.5040       750
weighted avg     0.5040    0.5040    0.5040       750
```

## Análise das Métricas

### Desempenho Geral

- **Acurácia**: ~50% (próximo ao acaso para problema binário)
- **F1-score**: ~0.50-0.51 (classe 1)
- **Log Loss**: ~0.693 (próximo de ln(2) ≈ 0.693, indicador de probabilidades não calibradas)

### Interpretação

!!! warning "Performance Limitada"
    Os resultados indicam que o modelo teve **dificuldade em capturar padrões discriminantes** nos dados. Isso pode ser explicado por:
    
    1. **Baixa correlação das features**: como observado na EDA, nenhuma variável individualmente apresenta correlação forte com Churn (máx ~0.02)
    2. **Dados simulados**: dataset pode ter sido gerado sinteticamente com pouca estrutura real
    3. **Complexidade do problema**: churn é inerentemente difícil de prever com apenas dados transacionais

### Comparação com Baseline

**Baseline (Majority Class)**:

- Prever sempre classe 0 (não churn): accuracy = 68.7%
- Prever sempre classe 1 (churn): accuracy = 31.3%

Nosso modelo:

- Accuracy = 50.4% (pior que baseline!)
- Porém, **F1-score balanceado** em ambas as classes
- Indica que o modelo está tentando generalizar, não apenas memorizar a classe majoritária

## Visualizações Disponíveis

O notebook `3_train_egc.ipynb` gera automaticamente:

1. **Matriz de Confusão** (heatmap)
2. **Curva ROC** com AUC
3. **Curva Precision-Recall** com Average Precision
4. **Distribuição de probabilidades** por classe

Essas visualizações são geradas via `plot_all()` em `train_mlp_cv_optuna.py`.

## Métrica da Competição

**Métrica oficial**: **F1-score**

A competição avalia com base no F1-score do conjunto de teste privado (50% das 2k amostras de teste).

Nosso F1-score no teste interno: **~0.51** (classe 1)
