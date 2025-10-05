# Conclusão

Este projeto aplicou MLPs (Multi-Layer Perceptrons) para predição de churn em um dataset de competição do Kaggle, explorando técnicas de pré-processamento, otimização de hiperparâmetros e avaliação robusta.

## Principais Resultados

### Performance do Modelo

- **Acurácia no teste**: ~50.4%
- **F1-score (classe 1)**: ~0.51
- **Log Loss**: ~0.693

### Arquitetura Final

- **Camadas ocultas**: [411, 359, 361] neurônios
- **Ativação**: logistic (sigmoid)
- **Solver**: LBFGS
- **Regularização L2**: α = 7.78e-05

## Principais Aprendizados

### 1. Qualidade dos Dados é Crucial

A **baixa correlação** entre features e target (máx ~0.02) limitou severamente a capacidade preditiva:

- Nenhuma variável individual mostrou poder discriminante forte
- MLP não conseguiu capturar interações não-lineares suficientemente úteis
- Sugere que o dataset pode ser sinteticamente gerado sem padrões reais

### 2. Balanceamento de Classes

O **split balanceado 70/15/15** foi essencial para:

- Métricas estáveis e confiáveis
- Evitar viés do modelo para a classe majoritária
- Avaliar verdadeira capacidade de generalização

### 3. Otimização de Hiperparâmetros

**Optuna** facilitou a busca automática:

- 10 trials foram suficientes para convergir
- LBFGS emergiu como solver mais robusto
- Regularização L2 preveniu overfitting

### 4. Importância do Pipeline Completo

Implementar todo o fluxo (split → CV → treino → validação → teste → retreino) ensinou:

- **Nunca avaliar no conjunto de teste** durante desenvolvimento
- **Cross-validation** fornece sinal de robustez
- **Retreinar em train+val** maximiza dados para submissão final

## Limitações

### 1. Performance do Modelo

- Acurácia de ~50% indica que o modelo não superou predição aleatória
- Log loss ~0.693 sugere probabilidades mal calibradas
- F1-score baixo em ambas as classes

### 2. Falta de Engenharia de Features

- Não criamos features derivadas (interações, agregados, razões)
- Não exploramos conhecimento de domínio específico de churn
- Usamos apenas as features fornecidas "as-is"

### 3. Modelos Alternativos Não Explorados

- Não comparamos com tree-based models (XGBoost, LightGBM, CatBoost)
- Não testamos ensembles ou stacking
- Não exploramos redes mais profundas ou arquiteturas especializadas

## Trabalhos Futuros

### Melhoria Imediata

1. **Feature Engineering**:

   - Criar razões: `Monthly_Spending / Account_Age_Months`
   - Interações: `Satisfaction_Score * Support_Calls`
   - Binning de variáveis numéricas

2. **Modelos Alternativos**:

   - XGBoost/LightGBM (geralmente superiores em dados tabulares)
   - Ensemble de MLP + árvores
   - Stacking com meta-learner

3. **Otimização de Threshold**:

   - Buscar threshold ótimo para maximizar F1
   - Calibração de probabilidades (Platt scaling, isotonic regression)

### Exploração Adicional

4. **Análise de Erros**:

   - Investigar casos mal classificados
   - Identificar subgrupos problemáticos
   - Criar features específicas para esses casos

5. **Dados Externos**:

   - Buscar datasets de churn reais para validação
   - Comparar padrões com literatura de churn

6. **Técnicas Avançadas**:

   - SMOTE/ADASYN para balanceamento sintético
   - Cost-sensitive learning
   - Focal Loss para lidar com desbalanceamento

## Considerações Finais

Embora a **performance absoluta** tenha sido limitada, o projeto foi extremamente valioso para:

- Compreender o **pipeline completo** de ML
- Implementar **split balanceado** e cross-validation corretamente
- Usar **Optuna** para otimização automática
- Praticar **documentação** completa do projeto
- Reconhecer quando **dados são o gargalo**, não o modelo

Em problemas reais de churn, espera-se F1-scores na faixa de 0.6-0.8 com bons dados e feature engineering adequado. Nosso resultado (~0.51) reflete as limitações do dataset fornecido.
