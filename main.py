import numpy as np
import random
import matplotlib.pyplot as plt

def kmeans(X, k, max_iterations=100):
    # Inicializa os centroides aleatoriamente
    centroids = X[np.random.choice(range(len(X)), k, replace=False)]
    for i in range(max_iterations):
        # Atribui cada ponto ao centroide mais próximo
        clusters = [[] for _ in range(k)]
        for x in X:
            distances = [np.linalg.norm(x - c) for c in centroids]
            cluster_index = np.argmin(distances)
            clusters[cluster_index].append(x)

        # Atualiza os centroides
        prev_centroids = centroids
        centroids = []
        for cluster in clusters:
            centroids.append(np.mean(cluster, axis=0))

        # Verifica se os centroides mudaram
        if set([tuple(c) for c in centroids]) == set([tuple(c) for c in prev_centroids]):
            break

    # Retorna os centroides e as atribuições dos pontos
    return np.array(centroids), clusters

# Exemplo de uso
# X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
X = np.random.random(size=(50, 2))
centroids, clusters = kmeans(X, k=3)
print(clusters)

# Plotagem dos resultados
colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
for i, cluster in enumerate(clusters):
    for x in cluster:
        ax.scatter(x[0], x[1], color=colors[i])
    ax.scatter(centroids[i][0], centroids[i][1], color=colors[i], marker='o', s=200, linewidth=3)
plt.show()
