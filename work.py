import csv
from collections import Counter
import matplotlib.pyplot as plt

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
# 5. PLOTAGEM DOS 6 GRÁFICOS
# ==============================

print("Gerando gráficos...")

# Criando uma matriz de gráficos: 3 linhas (Tipos de Gráfico) e 2 colunas (Entrada / Saída)
fig, axs = plt.subplots(3, 2, figsize=(12, 14))

cor_entrada = 'dodgerblue'
cor_saida = 'tomato'

# --- LINHA 1: HISTOGRAMAS ---
axs[0, 0].hist(in_degrees, bins=50, color=cor_entrada, edgecolor='black', alpha=0.7)
axs[0, 0].set_title('Histograma - Grau de Entrada')
axs[0, 0].set_xlabel('Grau')
axs[0, 0].set_ylabel('Frequência (Nº de Vértices)')
axs[0, 0].grid(True, axis='y', alpha=0.3)

axs[0, 1].hist(out_degrees, bins=50, color=cor_saida, edgecolor='black', alpha=0.7)
axs[0, 1].set_title('Histograma - Grau de Saída')
axs[0, 1].set_xlabel('Grau')
axs[0, 1].set_ylabel('Frequência (Nº de Vértices)')
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

# --- LINHA 3: DISTRIBUIÇÃO LOG-LOG ---
axs[2, 0].scatter(x_in, y_in, color=cor_entrada, alpha=0.6, edgecolors='none')
axs[2, 0].set_title('Distribuição Log-Log - Entrada')
axs[2, 0].set_xlabel('Grau (k)')
axs[2, 0].set_ylabel('P(k)')
axs[2, 0].set_xscale('log')
axs[2, 0].set_yscale('log')
axs[2, 0].grid(True, which="both", alpha=0.3)

axs[2, 1].scatter(x_out, y_out, color=cor_saida, alpha=0.6, edgecolors='none')
axs[2, 1].set_title('Distribuição Log-Log - Saída')
axs[2, 1].set_xlabel('Grau (k)')
axs[2, 1].set_ylabel('P(k)')
axs[2, 1].set_xscale('log')
axs[2, 1].set_yscale('log')
axs[2, 1].grid(True, which="both", alpha=0.3)

# Ajusta o espaçamento entre os gráficos para não sobrepor textos
plt.tight_layout()

# Exibe a janela com os gráficos
plt.show()

M47