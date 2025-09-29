# Dataset Selection

No momento de decisão do dataset utilizado nesse projeto, foi considerado a busca do grupo por uma maior interação com dados reais, que poderiam complexificar os desafios trazidos pelo dataset. Diante disso, foram explorados diversos datasets de desafios/competições presentes no [Kaggle](https://www.kaggle.com/competitions). Ao analisar algumas competições, foi escolhida a competição [Ultimate Customer Churn Prediction Challenge](https://www.kaggle.com/competitions/ultimate-customer-churn-prediction-challenge), por trazer um problema real do dia a dia de empresas do varejo - como lidar com o cancelamento de assinaturas dos clientes.

Além disso, o churn é um dos indicadores mais críticos para o varejo moderno, especialmente em modelos de assinatura e relacionamento contínuo com o cliente: ele corrói receita recorrente, reduz o LTV (lifetime value) e impede o payback do CAC (custo de aquisição), pressionando diretamente a rentabilidade. Em mercados de margem apertada e alta competição digital, pequenas variações na taxa de cancelamento geram grandes impactos no fluxo de caixa e no planejamento de demanda.

Em síntese, entendendo a importância desse indicador e por consequência a previsão dele, o dataset foi selecionado, buscando replicar o desafio do dia a dia de diversas varejistas através da competição.

Com relação ao dataset em si, ele conta com 17 colunas, sendo uma delas o target (*Churn*):

- Customer_ID
- Age
- Gender
- Location
- Subscription_Type
- Account_Age_Months
- Monthly_Spending
- Total_Usage_Hours
- Support_Calls
- Late_Payments
- Streaming_Usage
- Discount_Used
- Satisfaction_Score
- Last_Interaction_Type
- Complaint_Tickets
- Promo_Opted_In
- Churn

Contando com $8000$ linhas no sample de treino e $2000$ linhas no sample de teste utilizado para submissão da competição.