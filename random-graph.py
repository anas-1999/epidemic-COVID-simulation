import networkx as nx
import random
import math
import numpy as np
import scipy
import matplotlib.pyplot as plt

class Graph:
    def __init__(self, N, input_graph, transmision_probability = None, healing_probability = None):
        """Initialize all the variables and graph to be used by all operations"""
        self.N = N
        self.input_graph = input_graph.copy()
        self.virus_strength = 0.0
        self.transmision_probability = transmision_probability
        self.healing_probability = healing_probability
        self.infected = []
        self.susceptible = []
        self.c = 0

    def calculate_val_prop(self):
        """Fonction qui calcule valeur propre de la matrice d'adjacence
        """
        self.N = nx.number_of_nodes(self.input_graph)
        val_prop = scipy.sparse.linalg.eigsh(nx.to_numpy_matrix(self.input_graph),
            k=1, which='LM', return_eigenvectors=False, mode='normal')

        self.max_val_prop = max(val_prop)
        print("Max valeur propre value: " + str(self.max_val_prop))

    def calculate_virus_strength(self, transmision_probability, healing_probability):
        """ Fonction qui calcule l'intensité du virus"""
        self.transmision_probability = transmision_probability
        self.healing_probability = healing_probability
        K = self.transmision_probability / self.healing_probability
        self.virus_strength = self.max_val_prop * K
        return self.virus_strength

    def infect_initial(self):
        """ Infecte les noeuds initiales du graphe"""
        self.c = math.ceil(self.N/10)
        self.infected.clear()

        #Choisit c noeuds aléatoires du graphe pour être infecter. et être ajouter à la liste des noeuds inféctés
        self.infected = random.sample( nx.nodes(self.input_graph) , self.c )

    def get_susceptible(self):
        self.susceptible.clear()
        #Ajoute chaque voisins des neouds inféctés à la liste des suspects
      for node in self.infected:
            neighbours = self.input_graph.neighbors(node)
            self.susceptible.extend(neighbours)
        #Supprime tous les noeuds infectés de la liste des suspects
        self.susceptible = list( set(self.susceptible).difference(set(self.infected)) )

    def infect(self):
        for node in self.susceptible:
            #Choisir une variable dans l'intervalle [1,0]
            x = random.random()

            #Si x est inférieure à transmision_probability, on infecte le noeud
            if x <= self.transmision_probability:
                self.infected.append(node)

    def heal(self):
        for node in self.infected[:]:
            #Choisir une variable dans l'intervalle [1,0]
            x = random.random()
            #Si x est inférieure à healing_probability, on immunise le noeud, en le supprimant des listes des inféctés
            if x <= self.healing_probability:
                self.infected.remove(node)

    def get_fraction_of_infected_nodes(self):
        # Pourcentage de la population infectée
        return len(self.infected) / self.N

    def immune_policy_A(self, K):
        """Choisir k noeud pour l'immunisation."""
        nodes = random.sample( range(0,self.N) , K )
        #retirer les noeuds immunisés du graphe
        self.input_graph.remove_nodes_from(nodes)
        #réinitialiser le nombre des noeuds
        self.N = nx.number_of_nodes(self.input_graph)

    def immune_policy_B(self, K):
        """
        Sélectionner les k noeuds avec le plus grand degré pour l'immunisation
        """
        degree = nx.degree(self.input_graph)
        #Trier les noeuds en fonction du degré
        highest_degree_nodes  = [x for x,y in sorted(degree.items(), key=lambda x: x[1], reverse=True)]
        #retirer les noeuds immunisés du graphe
        self.input_graph.remove_nodes_from( highest_degree_nodes[:K] )
        self.N = nx.number_of_nodes(self.input_graph)

    def immune_policy_C(self, K):
        """Select the node with the highest degree for immunization. Remove this node (and its
incident edges) from the contact network. Repeat until all vaccines are administered."""
        for i in range(K):
            degree = nx.degree(self.input_graph)
            max_degree_node = max(degree.items(), key=lambda x: x[1])[0]
            #remove the immunized nodes from graph
            self.input_graph.remove_node( max_degree_node )
        self.N = nx.number_of_nodes(self.input_graph)

    def immune_policy_D(self, K):


        self.N = nx.number_of_nodes(self.input_graph)

        #Calcul des plus grandes valeurs et vecteurs propres
        eig_value, eig_vector = scipy.sparse.linalg.eigsh( nx.to_numpy_matrix(self.input_graph),
            k=1, which='LM', return_eigenvectors=True, mode='normal')
        eig_vector_list = sorted(
            [(a, abs(b)) for a, b in enumerate(eig_vector)], key = lambda x: x[1], reverse=True)

        #donne l'indice des k grandes valeurs de eig_vector_list
        nodes = [eig_vector_list[i][0] for i in range(K)]
        #retirer les noeuds immunisés du graphe
        self.input_graph.remove_nodes_from(nodes)
        self.N = nx.number_of_nodes(self.input_graph)

    def __repr__(self):
        """ A function for proper representation """
        return_str = "transmision_probability: " + str(self.transmision_probability)
        return_str += "\nhealing_probability: " + str(self.healing_probability)
        return_str += "\nvirus strength: " + str(self.virus_strength)
        return return_str

def read_input():
    """ Fonction qui crée un networkx graph"""
    graph_file = open('../data/static.network','r')
    nodes, edges = [int(x) for x in graph_file.readline().split(' ')]
    G = nx.Graph()
    for line in graph_file:
        a, b = [int(x) for x in line.split(' ')]
        G.add_edge(a,b)
    return nodes,edges,G

def plot_line_graphs(x, y, x1, y1):
    plt.plot( x, y, 'r')
    plt.xlabel(x1)
    plt.ylabel(y1)
    plt.show()
