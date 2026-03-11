import csv
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# ==============================
# IMPORT DA BIBLIOTECA ALGS4
# ==============================
from algs4.digraph import Digraph

# ==============================
# 1. LEITURA DO DATASET
# ==============================

print("Lendo dataset...")

node_map = {}
current_index = 0
edges = []

with open("act-mooc/mooc_actions.tsv", "r") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        user = "U_" + row["USERID"]
        target = "T_" + row["TARGETID"]

        if user not in node_map:
            node_map[user] = current_index
            current_index += 1

        if target not in node_map:
            node_map[target] = current_index
            current_index += 1

        edges.append((node_map[user], node_map[target]))

print("Construindo grafo...")

# ==============================
# 2. CONSTRUÇÃO DO GRAFO
# ==============================

G = Digraph(current_index)

for u, v in edges:
    G.add_edge(u, v)

V = G.V
E = G.E

# ==============================
# 3. CÁLCULO DOS GRAUS
# ==============================

print("Calculando graus de entrada e saída...")

# Grau de SAÍDA: Basta contar o tamanho da lista de adjacência de cada vértice
out_degrees = [len(list(G.adj[v])) for v in range(V)]

# Grau de ENTRADA: Precisamos varrer o grafo inteiro e contar as arestas que CHEGAM em cada vértice
in_degrees = [0] * V
for v in range(V):
    for w in list(G.adj[v]):
        in_degrees[w] += 1

# ==============================
# DEFINIÇÃO FORMAL E MÉTRICAS BÁSICAS
# ==============================
print("\n" + "="*60)
print("DEFINIÇÃO FORMAL DO GRAFO")
print("="*60)
print("Tipo: dirigido (dígrafo)")
print("Ponderado: não (todas as arestas têm peso unitário)")
print("Temporal: não (análise estática; dados originais têm timestamp)")
print()
print("MÉTRICAS")
print("-"*40)
print(f"Ordem |V| = {V}")
print(f"Tamanho |E| = {E}")
densidade = E / (V * (V - 1)) if V > 1 else 0.0
print(f"Densidade = |E|/(|V|*(|V|-1)) = {densidade:.6e}")
grau_medio_entrada = sum(in_degrees) / V
grau_medio_saida = sum(out_degrees) / V
print(f"Grau médio (entrada) = {grau_medio_entrada:.4f}")
print(f"Grau médio (saída)   = {grau_medio_saida:.4f}")
print("="*60 + "\n")

# ==============================
# 4. FUNÇÃO PARA DISTRIBUIÇÃO P(k)
# ==============================
# Essa função calcula P(k) ignorando grau 0 (importante para o gráfico log-log não quebrar)
def calc_distribuicao(lista_graus):
    graus_filtrados = [k for k in lista_graus if k > 0]
    n = len(graus_filtrados)
    contagem = Counter(graus_filtrados)
    
    x = sorted(contagem.keys())          # k (grau)
    y = [contagem[k] / n for k in x]     # P(k) (probabilidade)
    return x, y

x_in, y_in = calc_distribuicao(in_degrees)
x_out, y_out = calc_distribuicao(out_degrees)

# ==============================
# AJUSTE POR LEI DE POTÊNCIA P(k) ~ k^(-gamma)
# ==============================
# Justificativa metodológica: regressão linear em escala log-log (mínimos
# quadrados ordinários) para k >= k_min, para reduzir ruído na cauda esquerda
# (graus muito baixos desviam do comportamento em lei de potência).
K_MIN = 2

def ajuste_lei_potencia(x, y, k_min=K_MIN):
    mask = [i for i in range(len(x)) if x[i] >= k_min]
    if len(mask) < 3:
        return None, None, None
    log_k = np.log(np.array([x[i] for i in mask], dtype=float))
    log_p = np.log(np.array([y[i] for i in mask], dtype=float))
    coeffs = np.polyfit(log_k, log_p, 1)
    gamma = -coeffs[0]  # P(k) ~ k^(-gamma) => log P = -gamma*log k + c
    # R²
    pred = np.polyval(coeffs, log_k)
    ss_res = np.sum((log_p - pred) ** 2)
    ss_tot = np.sum((log_p - np.mean(log_p)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    return gamma, coeffs[1], r2

gamma_in, c_in, r2_in = ajuste_lei_potencia(x_in, y_in)
gamma_out, c_out, r2_out = ajuste_lei_potencia(x_out, y_out)

print("AJUSTE LEI DE POTÊNCIA P(k) ~ k^(-gamma)")
print("-"*50)
print("Método: regressão linear em log-log (MQO), k >= k_min =", K_MIN)
if gamma_in is not None:
    print(f"Grau de ENTRADA: gamma = {gamma_in:.3f}, R² = {r2_in:.4f}")
else:
    print("Grau de ENTRADA: insuficientes pontos para ajuste")
if gamma_out is not None:
    print(f"Grau de SAÍDA:   gamma = {gamma_out:.3f}, R² = {r2_out:.4f}")
else:
    print("Grau de SAÍDA:   insuficientes pontos para ajuste")
print()

# ==============================
# COEFICIENTE DE CLUSTERING MÉDIO (DÍGRAFO)
# ==============================
# Para cada vértice v: C(v) = (pares de vizinhos de saída conectados) / (total de pares).
# Dois vizinhos de saída a,b estão "conectados" se existe aresta (a,b) ou (b,a).
def clustering_medio_digrafo(G, V):
    total_c = 0.0
    count = 0
    adj = G.adj
    for v in range(V):
        vizinhos = list(adj[v])
        n = len(vizinhos)
        if n < 2:
            continue
        pares_conectados = 0
        for i in range(n):
            for j in range(i + 1, n):
                a, b = vizinhos[i], vizinhos[j]
                if b in adj[a] or a in adj[b]:
                    pares_conectados += 1
        c_v = pares_conectados / (n * (n - 1) / 2)
        total_c += c_v
        count += 1
    return total_c / count if count > 0 else 0.0

print("COEFICIENTE DE CLUSTERING MÉDIO")
print("-"*50)
clustering_medio = clustering_medio_digrafo(G, V)
print(f"Clustering médio (dígrafo, vizinhos de saída) = {clustering_medio:.6f}")
print()

# ==============================
# 5. PLOTAGEM DOS 6 GRÁFICOS
# ==============================

print("Gerando gráficos...")

# Criando uma matriz de gráficos: 3 linhas (Tipos de Gráfico) e 2 colunas (Entrada / Saída)
fig, axs = plt.subplots(3, 2, figsize=(12, 14))

cor_entrada = 'dodgerblue'
cor_saida = 'tomato'

max_in = max(in_degrees)
max_out = max(out_degrees)

# --- LINHA 1: HISTOGRAMAS ---
bins_in = range(0, max_in + 2, max(1, max_in // 50))
bins_out = range(0, max_out + 2, max(1, max_out // 50))

axs[0, 0].hist(in_degrees, bins=bins_in, color=cor_entrada, edgecolor='black', alpha=0.7)
axs[0, 0].set_title('Histograma - Grau de Entrada')
axs[0, 0].set_xlabel('Grau')
axs[0, 0].set_ylabel('Frequência (Nº de Vértices)')
axs[0, 0].set_yscale('log')
axs[0, 0].axvline(max_in, color='red', linestyle='--', linewidth=1)
axs[0, 0].text(max_in, 1.1, f"max={max_in}", rotation=90, va='bottom', ha='right', fontsize=8)
axs[0, 0].grid(True, axis='y', alpha=0.3)

axs[0, 1].hist(out_degrees, bins=bins_out, color=cor_saida, edgecolor='black', alpha=0.7)
axs[0, 1].set_title('Histograma - Grau de Saída')
axs[0, 1].set_xlabel('Grau')
axs[0, 1].set_ylabel('Frequência (Nº de Vértices)')
axs[0, 1].set_yscale('log')
axs[0, 1].axvline(max_out, color='red', linestyle='--', linewidth=1)
axs[0, 1].text(max_out, 1.1, f"max={max_out}", rotation=90, va='bottom', ha='right', fontsize=8)
axs[0, 1].grid(True, axis='y', alpha=0.3)

# --- LINHA 2: DISTRIBUIÇÃO LINEAR ---
# Usamos scatter (pontos) para distribuições P(k)
axs[1, 0].scatter(x_in, y_in, color=cor_entrada, alpha=0.6, edgecolors='none')
axs[1, 0].set_title('Distribuição Linear - Entrada')
axs[1, 0].set_xlabel('Grau (k)')
axs[1, 0].set_ylabel('P(k)')
axs[1, 0].grid(True, alpha=0.3)

axs[1, 1].scatter(x_out, y_out, color=cor_saida, alpha=0.6, edgecolors='none')
axs[1, 1].set_title('Distribuição Linear - Saída')
axs[1, 1].set_xlabel('Grau (k)')
axs[1, 1].set_ylabel('P(k)')
axs[1, 1].grid(True, alpha=0.3)

# --- LINHA 3: DISTRIBUIÇÃO LOG-LOG + AJUSTE LEI DE POTÊNCIA ---
axs[2, 0].scatter(x_in, y_in, color=cor_entrada, alpha=0.6, edgecolors='none', label='P(k)')
if gamma_in is not None:
    k_fit = np.array([k for k in x_in if k >= K_MIN], dtype=float)
    p_fit = np.exp(c_in) * np.power(k_fit, -gamma_in)
    axs[2, 0].plot(k_fit, p_fit, 'k--', linewidth=1.5, label=f'ajuste ~ k^{{-{gamma_in:.2f}}}')
axs[2, 0].set_title('Distribuição Log-Log - Entrada')
axs[2, 0].set_xlabel('Grau (k)')
axs[2, 0].set_ylabel('P(k)')
axs[2, 0].set_xscale('log')
axs[2, 0].set_yscale('log')
axs[2, 0].legend(loc='upper right', fontsize=8)
axs[2, 0].grid(True, which="both", alpha=0.3)

axs[2, 1].scatter(x_out, y_out, color=cor_saida, alpha=0.6, edgecolors='none', label='P(k)')
if gamma_out is not None:
    k_fit = np.array([k for k in x_out if k >= K_MIN], dtype=float)
    p_fit = np.exp(c_out) * np.power(k_fit, -gamma_out)
    axs[2, 1].plot(k_fit, p_fit, 'k--', linewidth=1.5, label=f'ajuste ~ k^{{-{gamma_out:.2f}}}')
axs[2, 1].set_title('Distribuição Log-Log - Saída')
axs[2, 1].set_xlabel('Grau (k)')
axs[2, 1].set_ylabel('P(k)')
axs[2, 1].set_xscale('log')
axs[2, 1].set_yscale('log')
axs[2, 1].legend(loc='upper right', fontsize=8)
axs[2, 1].grid(True, which="both", alpha=0.3)

# Ajusta o espaçamento entre os gráficos para não sobrepor textos
plt.tight_layout()

# Exibe a janela com os gráficos
plt.show()

print("Grau de entrada - min, max:", min(in_degrees), max(in_degrees))
print("Grau de saída  - min, max:", min(out_degrees), max(out_degrees))
print("Média grau entrada:", sum(in_degrees)/len(in_degrees))
print("Média grau saída:", sum(out_degrees)/len(out_degrees))

# ==============================
# DISCUSSÃO: REDE DE ESCALA LIVRE?
# ==============================
print("\n" + "="*60)
print("DISCUSSÃO: A REDE PODE SER CONSIDERADA DE ESCALA LIVRE?")
print("="*60)
print("""
Uma rede é frequentemente considerada 'de escala livre' quando:
 (1) A distribuição de graus P(k) segue aproximadamente uma lei de potência P(k) ~ k^(-gamma).
 (2) O expoente gamma costuma estar na faixa 2 < gamma < 3 em muitas redes reais.
 (3) Há forte assimetria: poucos hubs (alto grau) e muitos nós de grau baixo.

Interpretação para esta rede (MOOC actions):
 - Os valores de gamma e R² do ajuste (acima) indicam em que medida P(k) se aproxima
   de uma lei de potência. R² próximo de 1 sugere bom ajuste.
 - Se gamma está entre 2 e 3 e o ajuste é razoável, a rede exibe características
   de escala livre: poucos alvos muito populares (alto grau de entrada) e muitos
   nós com poucas conexões.
 - O coeficiente de clustering médio complementa: redes de escala livre muitas vezes
   têm clustering relativamente alto em relação a redes aleatórias.

Conclusão: Verifique os valores impressos de gamma e R². Se gamma ∈ (2, 3) e R²
aceitável, a rede pode ser considerada de escala livre nessa medida. Caso contrário,
discuta as limitações (cauda curta, cutoff, ou desvios na cauda).
""")
print("="*60)