# Dataset Explanation

A base de dados tabular, conforme já foi dito, conta com 8.000 clientes (linhas) e 17 colunas que descrevem perfil, plano/relacionamento e uso de um serviço por assinatura, como por exemplo SaaS/telecom/streaming. Aquie então, o objetivo é prever churn — se o cliente cancela ou não o serviço.

* **`Churn`** (inteiro binário):

  * **0** = não cancelou
  * **1** = cancelou

## Colunas de identificação

* **`Customer_ID`** (int): identificador único do cliente.

## Features (inputs) e tipos

**Categóricas (nominais)**

* **`Gender`** (objeto): {`Female`, `Male`}.
* **`Location`** (objeto): {`California`, `Florida`, `Illinois`, `New York`, `Texas`}.
* **`Subscription_Type`** (objeto): {`Basic`, `Premium`, `Enterprise`}.
* **`Last_Interaction_Type`** (objeto): {`Negative`, `Neutral`, `Positive`}.

**Categórica binária**

* **`Promo_Opted_In`** (int {0,1}): indica se aderiu a promoções.

**Numéricas (contínuas/discretas)**

* **`Age`** (int): idade; 18–69 (52 valores distintos).
* **`Account_Age_Months`** (int): tempo de conta em meses; 1–59.
* **`Monthly_Spending`** (float): gasto mensal; ≈ 10,09 a 199,94 (contínua).
* **`Total_Usage_Hours`** (int): horas totais de uso; 10–499.
* **`Support_Calls`** (int): número de ligações ao suporte; 0–9.
* **`Late_Payments`** (int): pagamentos em atraso; 0–4.
* **`Streaming_Usage`** (int): uso (unidade de consumo de streaming); 0–99.
* **`Discount_Used`** (int): descontos utilizados (unidade/índice); 0–99.
* **`Satisfaction_Score`** (int, **ordinal**): escore de satisfação; 1–10.
* **`Complaint_Tickets`** (int): chamados de reclamação; 0–4.

De modo geral, os dados não apresentam dados faltantes, provavelmente por ser uma competição. Claro que a não existência de dados faltantes diminui a complexidade de preparação dos dados, no entanto, no momento de previsão a dificuldade não é tão impactada por isso.

