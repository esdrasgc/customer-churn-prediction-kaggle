# Limpeza e Normalização dos Dados

Esta seção descreve como os dados foram preparados antes da modelagem.

## Visão Geral

- Classificação binária com variável target `Churn` (0 = cliente ativo, 1 = cliente cancelou).
- **Nenhum valor faltante** detectado no arquivo de treinamento fornecido pela competição.
- Implementamos pipeline robusto de pré-processamento para generalizar melhor e suportar inferência.

## Pipeline de Pré-processamento

Implementado em `notebooks/2_preprocess.ipynb` e `notebooks/train_mlp_cv_optuna.py` utilizando `ColumnTransformer` do scikit-learn:

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_vars),
        ('cat', OneHotEncoder(drop='first'), categorical_vars)
    ],
    remainder='passthrough'
)
```

### Transformações Aplicadas

- **Variáveis Numéricas**: `StandardScaler()` para normalização (média 0, desvio padrão 1)
- **Variáveis Categóricas**: `OneHotEncoder(drop='first')` para codificação one-hot (evita *dummy variable trap*)

## Tipagem das Features

Conforme definido em `notebooks/2_preprocess.ipynb`:

### Variáveis Categóricas
- `Gender` (Masculino/Feminino)
- `Location` (California, Florida, Illinois, New York, Texas)
- `Subscription_Type` (Basic, Premium, Enterprise)
- `Last_Interaction_Type` (Negative, Neutral, Positive)
- `Promo_Opted_In` (0/1 — tratado como categórico no pipeline)

### Variáveis Numéricas Contínuas
- `Age` (idade do cliente)
- `Account_Age_Months` (tempo de conta em meses)
- `Monthly_Spending` (gasto mensal)
- `Total_Usage_Hours` (horas totais de uso)
- `Streaming_Usage` (uso de streaming, 0-99%)
- `Discount_Used` (desconto utilizado, 0-99%)
- `Satisfaction_Score` (escore de satisfação, 1-10)

### Variáveis Numéricas Discretas (Contagens)
- `Support_Calls` (número de chamadas ao suporte)
- `Late_Payments` (número de pagamentos atrasados)
- `Complaint_Tickets` (número de tickets de reclamação)

## Balanceamento de Classes

- O dataset de treino apresenta desbalanceamento: ~68,7% (classe 0) vs ~31,3% (classe 1).
- Implementamos split estratificado 70/15/15 balanceado via `stratified_custom_split_indices` em `notebooks/train_mlp_cv_optuna.py`.
- Cada split (treino/validação/teste) contém **o mesmo número de positivos e negativos**, baseado na classe minoritária.

## Outliers e Duplicados

- Nenhuma remoção explícita de outliers foi aplicada.
- Os dados parecem limpos/simulados para uso na competição.
- Nenhuma duplicata foi reportada durante a EDA.

## Reprodutibilidade

- `random_state=42` fixado em todos os splits e inicializações de modelo.
- Dados pré-processados salvos em `data/X.pickle`, `data/y.pickle` e `data/X_teste.pickle`.
