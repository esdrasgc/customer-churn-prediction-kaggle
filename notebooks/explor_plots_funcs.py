from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns


def plot_target_distribution(df: pl.DataFrame, target: str = "Churn"):
    """Plota a distribuição da variável target"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Convertendo para pandas para facilitar os plots
    df_pd = df.to_pandas()

    # Contagem
    target_counts = df_pd[target].value_counts()
    axes[0].pie(
        target_counts.values,
        labels=["Não Churn", "Churn"],
        autopct="%1.1f%%",
        startangle=90,
    )
    axes[0].set_title(f"Distribuição de {target}")

    # Barplot
    sns.countplot(data=df_pd, x=target, ax=axes[1])
    axes[1].set_title(f"Contagem de {target}")
    axes[1].set_xlabel("Churn (0=Não, 1=Sim)")

    plt.tight_layout()
    plt.show()

    # Estatísticas
    churn_rate = df_pd[target].mean()
    print(f"Taxa de Churn: {churn_rate:.2%}")
    print(f"Total de clientes: {len(df_pd)}")
    print(f"Clientes que fizeram churn: {df_pd[target].sum()}")
    print(f"Clientes que permaneceram: {len(df_pd) - df_pd[target].sum()}")


def plot_categorical_analysis(
    df: pl.DataFrame, categorical_vars: List[str], target: str = "Churn"
):
    """Análise das variáveis categóricas vs target"""
    df_pd = df.to_pandas()
    n_vars = len(categorical_vars)
    n_cols = 2
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for i, var in enumerate(categorical_vars):
        # Crosstab para análise
        ct = pd.crosstab(df_pd[var], df_pd[target], normalize="index")
        ct.plot(kind="bar", ax=axes[i], rot=45)
        axes[i].set_title(f"{var} vs {target}")
        axes[i].set_ylabel("Proporção de Churn")
        axes[i].legend(["Não Churn", "Churn"])

        # Estatística qui-quadrado
        from scipy.stats import chi2_contingency

        chi2, p_value, _, _ = chi2_contingency(pd.crosstab(df_pd[var], df_pd[target]))
        axes[i].text(
            0.02,
            0.98,
            f"p-value: {p_value:.4f}",
            transform=axes[i].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # Remover eixos extras
    for i in range(len(categorical_vars), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


def plot_numerical_distributions(
    df: pl.DataFrame, numeric_vars: List[str], target: str = "Churn"
):
    """Plota distribuições das variáveis numéricas separadas por target"""
    df_pd = df.to_pandas()
    n_vars = len(numeric_vars)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for i, var in enumerate(numeric_vars):
        # Histograma separado por churn
        for churn_val in [0, 1]:
            data = df_pd[df_pd[target] == churn_val][var]
            axes[i].hist(data, alpha=0.6, label=f"Churn={churn_val}", bins=30)

        axes[i].set_title(f"Distribuição de {var}")
        axes[i].set_xlabel(var)
        axes[i].set_ylabel("Frequência")
        axes[i].legend()

        # Teste t para diferença de médias
        from scipy.stats import ttest_ind

        group0 = df_pd[df_pd[target] == 0][var]
        group1 = df_pd[df_pd[target] == 1][var]
        t_stat, p_value = ttest_ind(group0, group1)
        axes[i].text(
            0.02,
            0.98,
            f"p-value: {p_value:.4f}",
            transform=axes[i].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # Remover eixos extras
    for i in range(len(numeric_vars), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df: pl.DataFrame, numeric_vars: List[str]):
    """Plota matriz de correlação das variáveis numéricas"""
    df_pd = df.to_pandas()
    correlation_matrix = df_pd[numeric_vars].corr()

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Matriz de Correlação - Variáveis Numéricas")
    plt.tight_layout()
    plt.show()

    # Top correlações com churn
    if "Churn" in correlation_matrix.columns:
        churn_corr = correlation_matrix["Churn"].abs().sort_values(ascending=False)
        print("\nTop 10 correlações com Churn:")
        print(churn_corr.head(10))


def plot_boxplots_by_target(
    df: pl.DataFrame, numeric_vars: List[str], target: str = "Churn"
):
    """Boxplots das variáveis numéricas agrupadas por target"""
    df_pd = df.to_pandas()
    n_vars = len(numeric_vars)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for i, var in enumerate(numeric_vars):
        sns.boxplot(data=df_pd, x=target, y=var, ax=axes[i])
        axes[i].set_title(f"{var} por {target}")
        axes[i].set_xlabel("Churn (0=Não, 1=Sim)")

    # Remover eixos extras
    for i in range(len(numeric_vars), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


def plot_pairplot_sample(
    df: pl.DataFrame,
    vars_to_plot: List[str],
    target: str = "Churn",
    sample_size: int = 1000,
):
    """Pairplot de uma amostra dos dados"""
    df_pd = df.to_pandas()

    # Amostra para não sobrecarregar o plot
    if len(df_pd) > sample_size:
        df_sample = df_pd.sample(n=sample_size, random_state=42)
    else:
        df_sample = df_pd

    # Selecionar apenas algumas variáveis importantes
    vars_subset = vars_to_plot + [target]

    g = sns.pairplot(
        df_sample[vars_subset], hue=target, diag_kind="hist", plot_kws={"alpha": 0.6}
    )
    g.fig.suptitle("Pairplot - Variáveis Selecionadas", y=1.02)
    plt.show()
