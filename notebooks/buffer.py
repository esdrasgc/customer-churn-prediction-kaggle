# %% [markdown]
# Treinando MLP (scikit-learn) com GridSearch + holdout e avaliação completa
# - Explora arquiteturas, ativações, otimizadores, batch_size, L2 (alpha) e early stopping
# - Mantém pipeline de pré-processamento (imputação, OHE e padronização)
# - Salva resultados e modelo
# - (Opcional) NN do zero em NumPy para demonstrar o loop de treinamento

# %%
import os
import json
import math
import warnings
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import optuna
from optuna.integration import PyTorchLightningPruningCallback

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier  # Manter para compatibilidade
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_sample_weight  # Opcional: pode remover se não usar
from sklearn.exceptions import ConvergenceWarning
import pathlib

# =========================
# Configurações gerais
# =========================
RANDOM_STATE = 42
TEST_SIZE = 0.15    # 70/15/15 split
CV_SPLITS = 5
PRIMARY_SCORING_BINARY = "roc_auc"
PRIMARY_SCORING_MULTICLASS = "roc_auc_ovr"
path_data = pathlib.Path().cwd().parent / "data" 

DATA_PATH = path_data / 'train.csv'
TARGET: Optional[str] = "Churn"

# Configurações PyTorch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando device: {DEVICE}")

# Configurar para usar múltiplas GPUs se disponível
if torch.cuda.is_available():
    print(f"GPU disponível: {torch.cuda.get_device_name(0)}")
    print(f"Memória GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    # Otimizações para GPU
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# Configurações Optuna com paralelização
N_TRIALS = 100
EPOCHS_PER_TRIAL = 200
PATIENCE = 20
N_JOBS = -1  # Usar todos os cores disponíveis para Optuna

# Onde salvar artefatos
OUTPUT_DIR = "./pytorch_optuna_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configurar seeds para reprodutibilidade
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_STATE)
    torch.cuda.manual_seed_all(RANDOM_STATE)  # Para múltiplas GPUs

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# =========================
# Utilidades
# =========================
COMMON_TARGET_CANDIDATES = [
    "churn", "Churn", "CHURN",
    "target", "TARGET", "label", "Label", "y",
    "Exited", "is_churn", "default"
]

def infer_target_column(df: pd.DataFrame) -> str:
    """Tenta inferir a coluna-alvo:
    1) nomes comuns (acima);
    2) alguma coluna binária com nome sugestivo;
    3) por último, alguma coluna claramente binária.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in COMMON_TARGET_CANDIDATES:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]

    # procura por colunas com 2 valores únicos (binárias)
    binary_cols = []
    for c in df.columns:
        uniques = pd.Series(df[c].dropna().unique())
        if len(uniques) == 2:
            binary_cols.append(c)

    # heurística: se tiver 'churn' no nome
    for c in df.columns:
        if "churn" in c.lower():
            return c

    if len(binary_cols) == 1:
        return binary_cols[0]

    if len(binary_cols) > 1:
        # arbitrariamente pegue a primeira, mas avise
        print(f"[AVISO] Múltiplas colunas binárias candidatas: {binary_cols}. "
              f"Usando {binary_cols[0]}. Defina TARGET manualmente para garantir.")
        return binary_cols[0]

    raise ValueError(
        "Não foi possível inferir a coluna-alvo. "
        "Defina TARGET manualmente (ex.: TARGET = 'Churn')."
    )


def build_preprocessor(df: pd.DataFrame, target_col: str) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Cria um ColumnTransformer:
       - num: imputação mediana + padronização
       - cat: imputação modo + OneHotEncoder (dense)
    """
    # Se o alvo estiver no df, remova pra detectar features
    feature_df = df.drop(columns=[target_col]) if target_col in df.columns else df

    numeric_features = feature_df.select_dtypes(include=["number", "float", "int", "bool"]).columns.tolist()
    categorical_features = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()

    # OneHotEncoder compat (sklearn >=1.2 usa sparse_output)
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe)
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"
    )

    return preprocessor, numeric_features, categorical_features


def primary_scoring_for_target(y: pd.Series) -> str:
    """Define métrica principal de refit."""
    n_unique = y.nunique(dropna=True)
    if n_unique <= 2:
        return PRIMARY_SCORING_BINARY
    return PRIMARY_SCORING_MULTICLASS


# =========================
# Carregar dados
# =========================
df = pd.read_csv(DATA_PATH)
print(f"Shape do dataset: {df.shape}")

# Definir/Inferir target
if TARGET is None:
    TARGET = infer_target_column(df)
print(f"Coluna alvo: {TARGET}")

# Se o alvo vier como string/Yes/No etc., tente mapear para {0,1} se binário
y_raw = df[TARGET]
if y_raw.dtype == object:
    # Tenta mapear strings comuns para binário
    map_yes = {"yes": 1, "y": 1, "true": 1, "sim": 1, "churn": 1, "1": 1}
    map_no = {"no": 0, "n": 0, "false": 0, "nao": 0, "não": 0, "retained": 0, "0": 0}
    y_lower = y_raw.astype(str).str.lower().str.strip()
    mapped = y_lower.map(lambda v: 1 if v in map_yes else (0 if v in map_no else np.nan))
    # Se mapeou quase tudo, usa o mapeamento
    if mapped.notna().mean() > 0.9:
        y = mapped.astype(int)
    else:
        # Deixa como estava (pode ser multiclass ou stratification por string)
        y = y_raw.copy()
else:
    y = y_raw.copy()

X = df.drop(columns=[TARGET])


# =========================
# Split: holdout de teste
# =========================
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

print(f"Split -> train/val: {X_train_val.shape}, test: {X_test.shape}")

# =========================
# Rede Neural PyTorch
# =========================
class MLPNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.2, activation='relu'):
        super(MLPNet, self).__init__()
        
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)
        
        # Primeira camada
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        # Camada de saída
        self.output_layer = nn.Linear(prev_size, output_size)
        
        # Função de ativação
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Ativação {activation} não suportada")
    
    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
            x = self.dropout(x)
        x = self.output_layer(x)
        return x


# =========================
# Funções de treinamento e avaliação
# =========================
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, patience, device):
    """Treina o modelo com early stopping"""
    train_losses = []
    val_losses = []
    val_aucs = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Usar DataParallel se múltiplas GPUs disponíveis
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Usando {torch.cuda.device_count()} GPUs")
    
    for epoch in range(epochs):
        # Treinamento
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validação
        model.eval()
        val_loss = 0.0
        val_probs = []
        val_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                # Para calcular AUC
                if outputs.shape[1] == 1:  # binário
                    probs = torch.sigmoid(outputs).cpu().numpy()
                else:  # multiclass
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                
                val_probs.extend(probs)
                val_targets.extend(batch_y.cpu().numpy())
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Calcular AUC
        try:
            if len(np.unique(val_targets)) <= 2:
                if len(np.array(val_probs).shape) > 1 and np.array(val_probs).shape[1] > 1:
                    val_auc = roc_auc_score(val_targets, np.array(val_probs)[:, 1])
                else:
                    val_auc = roc_auc_score(val_targets, val_probs)
            else:
                val_auc = roc_auc_score(val_targets, val_probs, multi_class='ovr')
        except:
            val_auc = 0.0
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Salvar estado do modelo (considerando DataParallel)
            if hasattr(model, 'module'):
                best_model_state = model.module.state_dict().copy()
            else:
                best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Carregar melhor modelo
    if best_model_state is not None:
        if hasattr(model, 'module'):
            model.module.load_state_dict(best_model_state)
        else:
            model.load_state_dict(best_model_state)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_aucs': val_aucs,
        'best_val_loss': best_val_loss,
        'best_val_auc': max(val_aucs) if val_aucs else 0.0
    }


def objective(trial, X_train_processed, y_train_processed, X_val_processed, y_val_processed, input_size, output_size, device):
    """Função objetivo para o Optuna"""
    
    # Hiperparâmetros a serem otimizados
    hidden_layers = trial.suggest_int('hidden_layers', 1, 4)
    hidden_sizes = []
    for i in range(hidden_layers):
        size = trial.suggest_int(f'hidden_size_{i}', 32, 512, step=32)
        hidden_sizes.append(size)
    
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid'])
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'adamw'])
    
    # Criar modelo com pesos aleatórios (cada trial começa do zero)
    model = MLPNet(input_size, hidden_sizes, output_size, dropout_rate, activation).to(device)
    
    # Inicialização de pesos aleatória para cada trial
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    # Criar dataloaders com otimizações para GPU
    train_dataset = TensorDataset(X_train_processed, y_train_processed)
    val_dataset = TensorDataset(X_val_processed, y_val_processed)
    
    # Configurar DataLoaders com pin_memory e num_workers para GPU
    num_workers = 4 if device.type == 'cuda' else 0
    pin_memory = device.type == 'cuda'
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    
    # Configurar otimizador
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Critério de perda
    if output_size == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Treinar modelo
    results = train_model(model, train_loader, val_loader, optimizer, criterion, 
                         EPOCHS_PER_TRIAL, PATIENCE, device)
    
    # Limpar memória GPU após cada trial
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Retornar métrica para otimização (queremos maximizar AUC)
    return results['best_val_auc']

# =========================
# Preprocessamento e preparação dos dados
# =========================
preprocessor, num_cols, cat_cols = build_preprocessor(df, TARGET)

# Pré-processar os dados
X_train_val_processed = preprocessor.fit_transform(X_train_val)
X_test_processed = preprocessor.transform(X_test)

# Preparar labels
if y_train_val.nunique() <= 2:
    # Binário
    output_size = 1
    y_train_val_processed = y_train_val.values.astype(np.float32).reshape(-1, 1)
    y_test_processed = y_test.values.astype(np.float32).reshape(-1, 1)
else:
    # Multiclass - converter para índices
    output_size = y_train_val.nunique()
    label_mapping = {label: idx for idx, label in enumerate(y_train_val.unique())}
    y_train_val_processed = np.array([label_mapping[label] for label in y_train_val]).astype(np.int64)
    y_test_processed = np.array([label_mapping[label] for label in y_test]).astype(np.int64)

# Split interno para validação
X_train_proc, X_val_proc, y_train_proc, y_val_proc = train_test_split(
    X_train_val_processed, y_train_val_processed, 
    test_size=0.15, 
    stratify=y_train_val, 
    random_state=RANDOM_STATE
)

# Converter para tensores PyTorch com pin_memory para GPU
X_train_tensor = torch.FloatTensor(X_train_proc)
y_train_tensor = torch.FloatTensor(y_train_proc) if output_size == 1 else torch.LongTensor(y_train_proc.squeeze())

X_val_tensor = torch.FloatTensor(X_val_proc)
y_val_tensor = torch.FloatTensor(y_val_proc) if output_size == 1 else torch.LongTensor(y_val_proc.squeeze())

X_test_tensor = torch.FloatTensor(X_test_processed)
y_test_tensor = torch.FloatTensor(y_test_processed) if output_size == 1 else torch.LongTensor(y_test_processed.squeeze())

# Mover para GPU apenas quando necessário (dentro dos DataLoaders)
input_size = X_train_val_processed.shape[1]

print(f"Input size: {input_size}")
print(f"Output size: {output_size}")
print(f"Train shape: {X_train_tensor.shape}")
print(f"Val shape: {X_val_tensor.shape}")
print(f"Test shape: {X_test_tensor.shape}")

# =========================
# Otimização com Optuna
# =========================
print(f"\nIniciando otimização Optuna com {N_TRIALS} trials...")

# Configurar Optuna para usar paralelização
study = optuna.create_study(
    direction='maximize',  # Maximizar AUC
    study_name='pytorch_mlp_optimization',
    storage=f'sqlite:///{OUTPUT_DIR}/optuna_study.db',
    load_if_exists=True,
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),  # TPE é eficiente para paralelização
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)  # Pruning para acelerar
)

# Adicionar callback para mostrar progresso
def show_progress_callback(study, trial):
    if trial.value is not None:
        print(f"Trial {trial.number + 1}/{N_TRIALS} - AUC: {trial.value:.4f}")
    else:
        print(f"Trial {trial.number + 1}/{N_TRIALS} - PRUNED")

# Executar otimização com n_jobs=-1 para usar todos os cores
study.optimize(
    lambda trial: objective(trial, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, 
                          input_size, output_size, DEVICE),
    n_trials=N_TRIALS,
    callbacks=[show_progress_callback],
    n_jobs=N_JOBS,  # Usar todos os cores disponíveis
    show_progress_bar=True
)

print(f"\nMelhor trial: {study.best_trial.number}")
print(f"Melhor AUC: {study.best_value:.4f}")
print(f"Melhores hiperparâmetros:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# Salvar resultados do estudo
study_results = {
    'best_params': study.best_params,
    'best_value': study.best_value,
    'best_trial': study.best_trial.number,
    'all_trials': [
        {
            'number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'state': trial.state.name
        } for trial in study.trials
    ]
}

study_path = os.path.join(OUTPUT_DIR, "optuna_study_results.json")
with open(study_path, "w", encoding="utf-8") as f:
    json.dump(study_results, f, indent=2, ensure_ascii=False)
print(f"Resultados do estudo Optuna salvos em: {study_path}")

# =========================
# Treinamento do melhor modelo
# =========================
print("\nTreinando modelo final com melhores hiperparâmetros...")

best_params = study.best_params
hidden_sizes = []
for i in range(best_params['hidden_layers']):
    if f'hidden_size_{i}' in best_params:
        hidden_sizes.append(best_params[f'hidden_size_{i}'])

# Criar e treinar modelo final
best_model = MLPNet(
    input_size, 
    hidden_sizes, 
    output_size, 
    best_params['dropout_rate'], 
    best_params['activation']
).to(DEVICE)

# Inicialização aleatória
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

best_model.apply(init_weights)

# Criar dataset completo de treino/val para treinamento final
X_full_train = torch.FloatTensor(X_train_val_processed)
y_full_train = torch.FloatTensor(y_train_val_processed) if output_size == 1 else torch.LongTensor(y_train_val_processed.squeeze())

full_dataset = TensorDataset(X_full_train, y_full_train)

# Configurar DataLoader para treinamento final com otimizações
num_workers = 4 if DEVICE.type == 'cuda' else 0
pin_memory = DEVICE.type == 'cuda'

full_loader = DataLoader(
    full_dataset, 
    batch_size=best_params['batch_size'], 
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=num_workers > 0
)

# Configurar otimizador
if best_params['optimizer'] == 'adam':
    optimizer = optim.Adam(best_model.parameters(), 
                          lr=best_params['learning_rate'], 
                          weight_decay=best_params['weight_decay'])
elif best_params['optimizer'] == 'sgd':
    optimizer = optim.SGD(best_model.parameters(), 
                         lr=best_params['learning_rate'], 
                         weight_decay=best_params['weight_decay'], 
                         momentum=0.9)
elif best_params['optimizer'] == 'adamw':
    optimizer = optim.AdamW(best_model.parameters(), 
                           lr=best_params['learning_rate'], 
                           weight_decay=best_params['weight_decay'])

# Critério de perda
if output_size == 1:
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()

# Treinar modelo final com otimizações para GPU
final_train_losses = []
best_model.train()

# Usar DataParallel se múltiplas GPUs disponíveis
if torch.cuda.device_count() > 1:
    best_model = nn.DataParallel(best_model)
    print(f"Usando {torch.cuda.device_count()} GPUs para treinamento final")

with tqdm(total=EPOCHS_PER_TRIAL, desc="Treinamento Final") as pbar:
    for epoch in range(EPOCHS_PER_TRIAL):
        epoch_loss = 0.0
        for batch_x, batch_y in full_loader:
            # Transferir para GPU com non_blocking
            batch_x, batch_y = batch_x.to(DEVICE, non_blocking=True), batch_y.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = best_model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss /= len(full_loader)
        final_train_losses.append(epoch_loss)
        
        pbar.set_postfix({'loss': f'{epoch_loss:.4f}'})
        pbar.update(1)

# =========================
# Avaliação no conjunto de teste
# =========================
print("\nAvaliando modelo no conjunto de teste...")

best_model.eval()
with torch.no_grad():
    # Transferir dados de teste para GPU
    X_test_gpu = X_test_tensor.to(DEVICE, non_blocking=True)
    test_outputs = best_model(X_test_gpu)
    
    if output_size == 1:
        # Binário
        test_probs = torch.sigmoid(test_outputs).cpu().numpy().flatten()
        test_preds = (test_probs >= 0.5).astype(int)
        
        acc = accuracy_score(y_test, test_preds)
        roc = roc_auc_score(y_test, test_probs)
        ap = average_precision_score(y_test, test_probs)
        f1 = f1_score(y_test, test_preds)
        
    else:
        # Multiclass
        test_probs = torch.softmax(test_outputs, dim=1).cpu().numpy()
        test_preds = test_outputs.argmax(dim=1).cpu().numpy()
        
        # Mapear de volta para labels originais
        reverse_mapping = {idx: label for label, idx in label_mapping.items()}
        test_preds_labels = [reverse_mapping[pred] for pred in test_preds]
        
        acc = accuracy_score(y_test, test_preds_labels)
        roc = roc_auc_score(y_test, test_probs, multi_class='ovr') if len(np.unique(y_test)) > 2 else np.nan
        ap = np.nan  # Não implementado para multiclass
        f1 = f1_score(y_test, test_preds_labels, average='macro')

cm = confusion_matrix(y_test, test_preds if output_size == 1 else test_preds_labels)
report = classification_report(y_test, test_preds if output_size == 1 else test_preds_labels)

print("\n=== Métricas no conjunto de TESTE ===")
print(f"Accuracy : {acc:.4f}")
print(f"ROC AUC  : {roc:.4f}" if not np.isnan(roc) else "ROC AUC  : n/a")
print(f"AvgPrec  : {ap:.4f}" if not np.isnan(ap) else "AvgPrec  : n/a")
print(f"F1       : {f1:.4f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

# Salvar métricas
test_metrics = {
    "accuracy": acc,
    "roc_auc": None if np.isnan(roc) else roc,
    "average_precision": None if np.isnan(ap) else ap,
    "f1": f1,
    "confusion_matrix": cm.tolist(),
    "best_params": study.best_params,
    "best_optuna_score": study.best_value
}

metrics_path = os.path.join(OUTPUT_DIR, "test_metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(test_metrics, f, indent=2, ensure_ascii=False)
print(f"Métricas de teste salvas em: {metrics_path}")

# =========================
# Plotar curva de perda do treinamento final
# =========================
plt.figure(figsize=(10, 6))

# Plot da curva de perda do treinamento final
plt.subplot(1, 2, 1)
plt.plot(range(1, len(final_train_losses) + 1), final_train_losses, linewidth=2)
plt.xlabel("Época")
plt.ylabel("Loss")
plt.title("Curva de Perda - Treinamento Final")
plt.grid(True, alpha=0.3)

# Plot da evolução dos trials do Optuna
plt.subplot(1, 2, 2)
trial_values = [trial.value for trial in study.trials if trial.value is not None]
plt.plot(range(1, len(trial_values) + 1), trial_values, marker='o', alpha=0.7)
plt.xlabel("Trial")
plt.ylabel("AUC Validação")
plt.title("Evolução dos Trials - Optuna")
plt.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "training_curves.png")
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"Curvas de treinamento salvas em: {fig_path}")

# =========================
# Salvar modelo final
# =========================
model_path = os.path.join(OUTPUT_DIR, "best_pytorch_model.pth")
# Considerar DataParallel ao salvar o modelo
model_state_dict = best_model.module.state_dict() if hasattr(best_model, 'module') else best_model.state_dict()

torch.save({
    'model_state_dict': model_state_dict,
    'model_params': {
        'input_size': input_size,
        'hidden_sizes': hidden_sizes,
        'output_size': output_size,
        'dropout_rate': best_params['dropout_rate'],
        'activation': best_params['activation']
    },
    'best_params': study.best_params,
    'test_metrics': test_metrics,
    'preprocessor': preprocessor
}, model_path)
print(f"Modelo PyTorch salvo em: {model_path}")

# Informações do hardware utilizado
if torch.cuda.is_available():
    print(f"\n💻 Hardware utilizado:")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memória GPU utilizada: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"   Memória GPU máxima: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")

# =========================
# Resumo dos resultados
# =========================
print("\n" + "="*60)
print("RESUMO DOS RESULTADOS")
print("="*60)
print(f"Número de trials executados: {len(study.trials)}")
print(f"Melhor AUC (validação): {study.best_value:.4f}")
print(f"AUC no teste: {roc:.4f}" if not np.isnan(roc) else "AUC no teste: n/a")
print(f"Acurácia no teste: {acc:.4f}")
print(f"F1 no teste: {f1:.4f}")
print("\nMelhores hiperparâmetros:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
print("="*60)


# %% [markdown]
# -----------------------------------------------
# Implementação concluída com PyTorch + Optuna
# - Cada trial usa inicialização aleatória de pesos
# - Otimização de hiperparâmetros com 100 trials
# - Treinamento com early stopping
# - Avaliação completa no conjunto de teste
# -----------------------------------------------

print("\n🎉 Execução concluída com sucesso!")
print(f"📊 Resultados salvos em: {OUTPUT_DIR}")
print("📈 Verifique os arquivos gerados para análise detalhada dos resultados.")
