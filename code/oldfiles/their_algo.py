import networkx as nx
import numpy as np
import scipy
import random
import matplotlib.pyplot as plt

def generate_erdos_renyi_graph(n, p, draw=False):
    # Generate an Erdős-Rényi graph with n nodes and edge probability p
    G = nx.erdos_renyi_graph(n, p)

    # Draw the graph
    if draw:
        plt.figure(figsize=(8, 8))
        nx.draw(G, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500, font_size=10)
        plt.title(f"Erdős–Rényi Graph (n={n}, p={p})")
        plt.show()
    
    return G


def algo5(n, gamma, A, G):
    def algo4(n, alpha_1, A, G):
        S_ast = algo2(n, alpha_1, A, G)
        p_Sf = algo3(n, alpha_1, A, S_ast, G)
        return p_Sf

    def algo3(n, alpha_1, A, S_ast, G):
        S_subgraph = G.subgraph(S_ast)
        p_S = [S_subgraph.degree(i) / len(S_ast) for i in S_subgraph.nodes]
        p_S_ast = np.mean(p_S)
        
        scores = {i: np.abs(p_S[i] - p_S_ast) for i in range(S_subgraph.number_of_nodes())}
        sorted_scores = sorted(scores, key=lambda x: scores[x])

        S_f = sorted_scores[:int(3 * alpha_1 * n)]
        p_Sf = np.mean([p_S[i] for i in S_f])

        return p_Sf

    def algo2(n, alpha_1, A, G):
        def get_p(S_subgraph):
            p_S = nx.average_clustering(S_subgraph)
            return p_S
        
        def metric(S_subgraph, A_S):
            p_S = get_p(S_subgraph)
            return np.linalg.norm(A_S - p_S, ord=2)
            
        S_comp = []
        S = np.arange(n)
        S_subgraph = G
        candidates = []
        for _ in range(1, int(9 * alpha_1 * n)):
            A_S = nx.adjacency_matrix(S_subgraph).toarray()
            _, eigenvectors = scipy.linalg.eigh(A_S, subset_by_index=[n-1, n-1])
            top_eigenvector = eigenvectors[:, -1]
            
            top_eigenvector_squared = top_eigenvector ** 2
            probabilities = top_eigenvector_squared / top_eigenvector_squared.sum()

            selected_index = np.random.choice(S_subgraph.nodes(), p=probabilities)
            
            S_subgraph.remove_node(selected_index)
            S_comp.append(selected_index)
            n -= 1

            candidates.append((S_comp, metric(S_subgraph, A_S)))
            
        S_comp, _ = min(candidates, key=lambda x: x[1])
        return np.delete(S, S_comp)
    p_ast = algo4(n, gamma, A, G)
    if p_ast <= 0.5:
        p_hat = p_ast
    else:
        q_ast = algo4(n, gamma, np.ones_like(A, dtype=float) - np.eye(n) - A, G)
        p_hat = 1 - q_ast
    return p_hat 

def perturb_graph(G, epsilon, p):
    """
    Perturbs the graph G by adding edges with probability p between an epsilon-fraction of nodes.

    Parameters:
    - G: networkx.Graph - The original graph.
    - epsilon: float - Fraction of nodes to perturb (0 < epsilon <= 1).
    - p: float - Probability of adding an edge between two nodes (0 <= p <= 1).

    Returns:
    - networkx.Graph - The perturbed graph.
    """
    # Create a copy of the graph to avoid modifying the original
    perturbed_G = G.copy()
    
    # Calculate the number of nodes to perturb
    n = len(G.nodes)
    perturb_count = int(epsilon * n)
    
    # Convert nodes to a list before sampling
    nodes_list = list(G.nodes)
    
    # Randomly sample epsilon-fraction of nodes
    nodes_to_perturb = random.sample(nodes_list, perturb_count)
    
    # For each selected node, decide to add edges with probability p
    for u in nodes_to_perturb:
        for v in G.nodes:
            if u != v:  # avoid self-loops and existing edges
                if random.random() < p:
                    if not perturbed_G.has_edge(u, v):
                        perturbed_G.add_edge(u, v)
                elif perturbed_G.has_edge(u, v):
                    perturbed_G.remove_edge(u, v)
    return perturbed_G

def perturb_graph_general(G, epsilon):
    """
    Perturbs the graph G by adding edges with probability p between an epsilon-fraction of nodes.

    Parameters:
    - G: networkx.Graph - The original graph.
    - epsilon: float - Fraction of nodes to perturb (0 < epsilon <= 1).
    - p: float - Probability of adding an edge between two nodes (0 <= p <= 1).

    Returns:
    - networkx.Graph - The perturbed graph.
    """
    # Create a copy of the graph to avoid modifying the original
    perturbed_G = G.copy()
    
    # Calculate the number of nodes to perturb
    n = len(G.nodes)
    perturb_count = int(epsilon * n)
    
    # Convert nodes to a list before sampling
    nodes_list = list(G.nodes)
    
    # Randomly sample epsilon-fraction of nodes
    nodes_to_perturb = random.sample(nodes_list, perturb_count)
    
    # For each selected node, decide to add edges with probability p
    for u in nodes_to_perturb:
        for v in G.nodes:
            if u != v:  # avoid self-loops and existing edges
                p = random.random()
                if random.random() < p:
                    if not perturbed_G.has_edge(u, v):
                        perturbed_G.add_edge(u, v)
                elif perturbed_G.has_edge(u, v):
                    perturbed_G.remove_edge(u, v)
    perturbed_L = np.array(nx.laplacian_matrix(perturbed_G).toarray())
    return perturbed_G, perturbed_L

def robust_estimate_guess(G, epsilon):
    L = np.array(nx.laplacian_matrix(G).toarray())
    D = np.diag(L)
    n = len(D)
    mean = np.mean(D) / n
    med = np.median(D) / n
    y = (1 / (1 - epsilon)) * abs(mean - med)
    # y = abs(mean - med) / (1 - epsilon)
    # y = abs(mean - med) / (math.e ** (-epsilon))
    if med <= mean:
        return med - y
    else:
        return med + y

def robust_estimation_using_var(G, epsilon):
    n = G.number_of_nodes()
    num_removals = int(epsilon * n)

    modified_G = G.copy()
    for i in range(num_removals):
        nodes = list(modified_G.nodes)
        sample_variances = {}
        L = nx.laplacian_matrix(modified_G).toarray()
        D = np.diag(L)
        for i, node in enumerate(nodes):
            modified_diag = np.copy(D)
            for j in range(len(D)):
                modified_diag[j] -= L[i][j]
            p_hat = np.mean(modified_diag) / (len(L) - 1)
            sample_variances[node] = abs(np.var(modified_diag, ddof=1) - (len(L) - 1) * p_hat * (1 - p_hat))

        # Find the node with the highest sample variance
        node_to_remove = min(sample_variances, key=sample_variances.get)
        modified_G.remove_node(node_to_remove)
    
    L = np.array(nx.laplacian_matrix(modified_G).toarray())
    D = np.diag(L)
    return np.mean(D) / modified_G.number_of_nodes()

def robust_estimation_using_var_prime(G, epsilon):
    n = G.number_of_nodes()
    def pruned_mean(d):
        # prune d
        d.sort()
        d = d[int((epsilon * n)//2) : int(-((epsilon * n)//2))]
        return np.mean(d)
    num_removals = int(epsilon * n)

    modified_G = G.copy()
    for i in range(num_removals):
        nodes = list(modified_G.nodes)
        sample_variances = {}
        L = nx.laplacian_matrix(modified_G).toarray()
        D = np.diag(L)
        for i, node in enumerate(nodes):
            modified_diag = np.copy(D)
            for j in range(len(D)):
                modified_diag[j] -= L[i][j]
            p_hat = pruned_mean(modified_diag) / (len(L) - epsilon * n - 1)
            sample_variances[node] = abs(np.var(modified_diag, ddof=1) - (len(L) - 1) * p_hat * (1 - p_hat))

        # Find the node with the highest sample variance
        node_to_remove = min(sample_variances, key=sample_variances.get)
        modified_G.remove_node(node_to_remove)
    
    L = np.array(nx.laplacian_matrix(modified_G).toarray())
    D = np.diag(L)
    return np.mean(D) / modified_G.number_of_nodes()


def test_q_adversary(n, gamma, p, q, algo='algo5'):
    G = generate_erdos_renyi_graph(n, p)
    p_original = nx.average_clustering(G)

    preturbed_G = perturb_graph(G, gamma, q)
    preturbed_A = nx.adjacency_matrix(preturbed_G).toarray()


    if algo == 'algo5':
        return p_original, algo5(n, gamma, preturbed_A, preturbed_G)
    elif algo == 'guess':
        return p_original, robust_estimate_guess(preturbed_G, gamma)
    elif algo == 'var':
        return p_original, robust_estimation_using_var(preturbed_G, gamma)
    elif algo == 'algo5var':
        return p_original, [robust_estimation_using_var(preturbed_G, gamma), algo5(n, gamma, preturbed_A, preturbed_G)]
    else:
       return p_original, np.array([
        algo5(n, gamma, preturbed_A, preturbed_G),
        robust_estimate_guess(preturbed_G, gamma),
        robust_estimation_using_var(preturbed_G, gamma)[0]
    ]) 


def varying_gamma_test():
    n = 100
    p = 0.3
    q = 0.8
    trials = 7

    algo5_errors = []
    var_errors = []
    guess_errors = []
    gammas = np.arange(0.01, 0.2, 0.01)
    for gamma in gammas:
        algo5_error = 0
        var_error = 0
        guess_error = 0
        for _ in range(trials):
            p_original, p_hat = test_q_adversary(n, gamma, p, q, algo='all')
            algo5_error += (np.abs(p_hat[0] - p_original) / p_original)
            guess_error += (np.abs(p_hat[1] - p_original) / p_original)
            var_error += (np.abs(p_hat[2] - p_original) / p_original)
        algo5_errors.append(algo5_error)
        var_errors.append(var_error)
        guess_errors.append(guess_error)

    
    plt.plot(gammas, algo5_errors, label="Algo 5")
    plt.plot(gammas, guess_errors, label="Guess")
    plt.plot(gammas, var_errors, label="Variance")
    plt.xlabel("Gamma")
    plt.ylabel("Relative Error")
    plt.title("Relative Error vs. Gamma")
    plt.legend()
    plt.show()

def varying_n_test():
    p = 0.3
    q = 0.6
    trials = 10

    algo5_errors = []
    var_errors = []
    # guess_errors = []
    ns = np.arange(500, 5000, 500)
    for n in ns:
        # algo5_error = 0
        var_error = 0
        # guess_error = 0
        for i in range(trials):
            # print(f"n: {n}, trial: {i}")
            p_original, p_hat = test_q_adversary(n, 0.1, p, q, algo='var')
            # if n < 200:
            #     algo5_error += (np.abs(p_hat[1] - p_original) / p_original)
            # else:
            #     algo5_error += 0
            var_error += (np.abs(p_hat - p_original) / p_original)
        # algo5_errors.append(algo5_error / trials)
        var_errors.append(var_error / trials)
        print(f"n: {n}, var_error: {var_error / trials}")
        # guess_errors.append(guess_error)
    

    plt.plot(ns, algo5_errors, label="Algo 5")
    # plt.plot(ns, guess_errors, label="Guess")
    plt.plot(ns, var_errors, label="Variance")
    plt.xlabel("n")
    plt.ylabel("Relative Error")
    plt.title("Relative Error vs. n")
    plt.legend()
    plt.show()
    
def compare_spectral_and_variance():
    n = 100
    p = 0.3
    q = 0.8
    trials = 7

    spectral_errors = []
    var_errors = []
    gammas = np.arange(0.01, 0.2, 0.01)
    for gamma in gammas:
        spectral_error = 0
        var_error = 0
        for _ in range(trials):
            p_original, p_hat = test_q_adversary(n, gamma, p, q, algo='algo5')
            spectral_error += (np.abs(p_hat[0] - p_original) / p_original)
            var_error += (np.abs(p_hat[1] - p_original) / p_original)
        spectral_errors.append(spectral_error)
        var_errors.append(var_error)

    
    plt.plot(gammas, spectral_errors, label="Spectral")
    plt.plot(gammas, var_errors, label="Variance")
    plt.xlabel("Gamma")
    plt.ylabel("Relative Error")
    plt.title("Relative Error vs. Gamma")
    plt.legend()
    plt.show()

def main():
    # varying_gamma_test()
    # varying_n_test()
    n = 500
    p = 0.3
    q = 0.8
    trials = 1

    var_errors = []
    gammas = np.arange(0.02, 0.2, 0.04)
    for gamma in gammas:
        var_error = 0
        for _ in range(trials):
            G = generate_erdos_renyi_graph(n, p)
            p_original = nx.average_clustering(G)

            preturbed_G = perturb_graph(G, gamma, q)
            
            p_hat = robust_estimation_using_var_prime(preturbed_G, gamma)
            var_error += (np.abs(p_hat - p_original))
            print(f"gamma: {gamma:.2f}\np:\t{p_original}\np_hat:\t{p_hat}\nerror:\t{var_error}\n")
        var_errors.append(var_error / trials)


    plt.plot(gammas, var_errors, label="Variance")
    plt.xlabel("Gamma")
    plt.ylabel("Relative Error")
    plt.title("Relative Error vs. Gamma")
    plt.legend()
    plt.show()
    


if __name__ == "__main__":
    main()