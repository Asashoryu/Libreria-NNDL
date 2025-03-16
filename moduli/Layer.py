import numpy as np


class Layer:
    """
    Classe per la creazione e gestione di layer di una rete neurale artificiale.

    Args:
        input_size (int): Numero di nodi del layer precedente.
        n_neurons (int): Numero di nodi desiderati.
        activation_function (callable): Funzione di attivazione scelta.
        derivative_activation (callable): Derivata della funzione di attivazione scelta.

    Attributes:
        input_size (int): Numero di nodi del layer precedente.
        n_neurons (int): Numero di nodi.
        activation_function (callable): Funzione di attivazione.
        derivative_activation (callable): Derivata della funzione di attivazione.
        weights (np.ndarray): Pesi del layer, matrice di dimensione (input_size x n_neurons).
        bias (np.ndarray): Bias del layer, vettore di dimensione (1 x n_neurons).
        unactived_output (np.ndarray): Output inattivato del layer corrente in forma matriciale.
        output (np.ndarray): Output finale del layer corrente in forma matriciale.
    """

    def __init__(self, input_size, n_neurons, activation_function, derivative_activation):
        self.input_size = input_size
        self.n_neurons = n_neurons
        self.activation_function = activation_function
        self.derivative_activation = derivative_activation
        # Genera una matrice casuale dei pesi
        self.weights = np.random.randn(input_size, n_neurons)
        # Genero un vettore di bias di zeri
        self.bias = np.zeros((1, n_neurons))

    def forward_pass(self, input):
        """
        Calcola l'output del layer corrente, immagine dell'input in ingresso.

        Args:
            input (np.ndarray): Dati in ingresso in forma matriciale, di dimensione (n x input_size).

        Returns:
            output (np.ndarray): Immagine dell'input in forma matriciale, di dimensione (n x n_neurons).
        """

        # L'output inattivato 'a' è ottenuto come il prodotto riga per colonna tra matrice input e matrice dei pesi, a cui sommiamo il bias uniformemente
        # Il bias è sommato ad ogni riga mediante broadcasting automatico
        a = np.dot(input, self.weights) + self.bias
        # L'output inattivato servirà come input per la derivata della funzione di attivazione nel calcolo dei delta interni
        self.unactived_output = a
        # L'output è l'immagine dell'output inattivato mediante la funzione di attivazione
        self.output = self.activation_function(a)
        return self.output

    def get_weights(self):
        """
        Restituisce una copia della matrice dei pesi del layer.

        Returns:
            (np.ndarray): Copia della matrice dei pesi.
        """

        return self.weights.copy()

    def set_weights(self, new_weights):
        """
        Setter per la matrice dei pesi utilizzando una copia della matrice in input.

        Args:
            new_weights (np.ndarray): Matrice dei pesi in input da settare.

        Raises:
            ValueError: se la forma della matrice in input non è (input_size x n_neurons)
        """

        if new_weights.shape != (self.input_size, self.n_neurons):
            raise ValueError(
                "La forma del vettore di pesi deve essere (input_size, n_neurons)")
        # Per evitare di modificare erroneamente la matrice dei pesi in input, ne facciamo una copia
        self.weights = new_weights.copy()

    def get_bias(self):
        """
        Restituisce una copia del vettore dei bias del layer.

        Returns:
            (np.ndarray): Copia del vettore dei bias.
        """

        return self.bias.copy()

    def set_bias(self, new_bias):
        """
        Setter per il vettore dei bias utilizzando una copia del vettore in input.

        Args:
            new_bias (np.ndarray): Vettore dei bias in input da settare.

        Raises:
            ValueError: se la forma del vettore in input non è (1 x n_neurons)
        """

        if new_bias.shape != (1, self.n_neurons):
            raise ValueError(
                "La forma del vettore di bias deve essere (1, n_neurons)")
        # Per evitare di modificare erroneamente il vettore dei bias in input, ne facciamo una copia
        self.bias = new_bias.copy()

    def get_input_size(self):
        """
        Getter per l'attributo input_size.

        Returns:
            input_size (int).
        """
        return self.input_size

    def get_n_neurons(self):
        """
        Getter per l'attributo n_neurons.

        Returns:
            n_neurons (int).
        """
        return self.n_neurons
