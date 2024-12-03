import nengo
import nengo_loihi
import numpy as np
import networkx as nx
import matplotlib as plt
import heapq
import collections
# from scipy.stats import binom
# from scipy.stats import truncnorm
# from scipy.stats import norm
import math
import types

from .utils import TopKHebbianNode

# Configurable assembly model for simulations
# Author Yau-Meng Wong, 2024

class Area:
  """A brain area.

  Attributes:
    name: the area's name (symbolic tag).
    model: nengo.Network.
    n: number of neurons in the area.
    k: number of neurons that fire in this area.
    p: probability of connection between two nodes.
    beta: Default value for activation-`beta`.
    w: Number of neurons that has ever fired in this area.
  """
  def __init__(self, name, model, n, k, p, *,
               beta=0.05, w=0):
    """Initializes the instance.

    Args:
      name: Area name (symbolic tag), must be unique.
      n: number of neurons
      k: number of firing neurons when activated.
      p: probability of connection
      beta: default activation-beta.
      w: initial 'winner' set-size.
    """
    self.name = name
    self.n = n
    self.k = k
    self.p = p
    self.beta = beta
    self.w = w

    # Create the Erdős–Rényi graph
    G = nx.erdos_renyi_graph(n, p)

    # Adjacency matrix of the graph
    adj_matrix = nx.adjacency_matrix(G).toarray()

    # Scale the weights to represent synaptic strengths
    weights = adj_matrix * 0.1  # Adjust scaling factor as needed

    with model:
        # Create an ensemble of neurons
        self.ensemble = nengo.Ensemble(
            n_neurons=n,
            dimensions=1,  # Single-dimensional representation
            neuron_type=nengo.LIF(),  # Use LIF neurons
        )

        # Create a Node to enforce the top-k constraint

        # Instantiate the node with initial weights, k, and learning rate
        top_k_node = TopKHebbianNode(weights, k, self.beta)

        # Create the Node
        top_k = nengo.Node(
            output=top_k_node,
            size_in=n,   # Receive input from all neurons
            size_out=n,  # Output modified inputs to neurons
        )

        # Connect ensemble.neurons to the Node
        nengo.Connection(
            self.ensemble.neurons,  # Source: neuron outputs (spikes)
            top_k,             # Destination: Node
            synapse=0.01,      # Synaptic filtering
        )

        # Connect the Node back to ensemble.neurons
        nengo.Connection(
            top_k,             # Source: Node output
            self.ensemble.neurons,  # Destination: neuron inputs
            synapse=0.01,      # Synaptic filtering
        )

        # Add a probe to record neural activities (filtered spikes)
        self.rate_probe = nengo.Probe(self.ensemble.neurons, synapse=0.01)

        # Add a probe to record spikes (raw spikes)
        self.spike_probe = nengo.Probe(self.ensemble.neurons, synapse=None)

    def plot_neural_activity():
        """Plots neural activity."""
        # Plot average neural activity (from rate_probe)
        plt.figure(figsize=(10, 5))
        plt.plot(sim.trange(), sim.data[self.rate_probe].mean(axis=1))
        plt.xlabel("Time (s)")
        plt.ylabel("Average Neural Activity (Hz)")
        plt.title(f"Erdős–Rényi Connectivity with Top-{k} Constraint in Nengo")
        plt.show()

    def plot_firing_neurons():
        """Plots the number of neurons firing at each timestep."""
        # Calculate the number of neurons firing at each timestep
        num_firing = np.sum(sim.data[self.spike_probe] > 0, axis=1)

        # Plot the number of neurons firing at each timestep
        plt.figure(figsize=(10, 5))
        plt.plot(sim.trange(), num_firing)
        plt.xlabel("Time (s)")
        plt.ylabel("Number of Neurons Firing")
        plt.title("Number of Neurons Firing at Each Timestep")
        plt.show()

def test_area():
    model = nengo.Network()

    # Create a brain area
    area = Area('new', model, 100, 10, 0.1)

    # Simulate the model
    with nengo_loihi.Simulator(model) as sim:
        sim.run(0.01)  # Simulate for 0.01 second

    area.plot_neural_activity()
    area.plot_firing_neurons()

test_area()