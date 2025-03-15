import csv
import time
import random
import sklearn.model_selection as sk
import scipy.special as sc
import numpy as np
import random

from funzioni_attivazione import sigmoid, der_sigmoid, identity, der_identity, tanh, der_tanh, relu, der_relu
from funzioni_errore import cross_entropy_with_softmax, cross_entropy_der_with_softmax, cross_entropy_binary, cross_entropy_binary_der, sum_of_squares, sum_of_squares_der
from plots import plot_fold_errors, plot_fold_accuracies, plot_epochs_errors
from data_processing import one_hot_encode, resize_mnist, load_mnist_784


class NeuralNetwork:
    """
    Classe per la creazione e gestione di una rete neurale artificiale.

    Args:
        input_size (int): Numero di feature.
        learning_rate (float): Tasso di apprendimento per l'aggiornamento dei pesi.
        error_function_name (str, optional): Funzione di errore da utilizzare ('cross_entropy' o 'sum_of_squares'). Default: 'cross_entropy'.
        softmax (bool, optional): Indica se applicare la softmax per problemi di classificazione multiclasse. Default: True.
        patience (int, optional): Numero massimo di epoche senza miglioramenti prima dello stop del training. Default: 10.

    Attributes:
        layers (list): Contiene i livelli della rete neurale.
        input_size (int): Numero di feature.
        learning_rate (float): Tasso di apprendimento.
        error_function_name (str, optional): Nome della funzione di errore utilizzata.
        softmax (bool): Flag che indica se la softmax è attiva.
        rprop_active (bool): Flag per indicare se l'algoritmo Rprop è attivo.
        patience (int): Numero massimo di epoche senza miglioramenti da eseguire.
        error_function (callable): Funzione di errore scelta.
        error_function_derivative (callable): Derivata della funzione di errore scelta.
      """

    def __init__(self, input_size, learning_rate, error_function_name='cross_entropy', softmax=True, patience=10):
        self.layers = []
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.error_function_name = error_function_name
        self.softmax = softmax
        self.rprop_active = False
        self.patience = patience
        # Configurazione della funzione di errore e della sua derivata in base alla scelta dell'utente
        if error_function_name == 'cross_entropy':
            if softmax:
                # Se il flag della softmax è attivo, si usa la cross-entropy con softmax
                self.error_function = cross_entropy_with_softmax
                self.error_function_derivative = cross_entropy_der_with_softmax
            else:
                # Se il flag della softmax non è attivo, si usa la cross-entropy binaria per problemi a due classi
                self.error_function = cross_entropy_binary
                self.error_function_derivative = cross_entropy_binary_der
        elif error_function_name == 'sum_of_squares':
            self.error_function = sum_of_squares
            self.error_function_derivative = sum_of_squares_der
        else:
            # Solleva un'eccezione se la funzione di errore non è supportata
            raise ValueError(
                f"Funzione di errore '{error_function_name}' non valida. Le opzioni valide sono 'cross_entropy' o 'sum_of_squares'.")

    def set_rprop(self, flag, eta_minus=0.5, eta_plus=1.2, delta_min=1e-6, delta_max=50):
        """
        Attiva o disattiva l'algoritmo Rprop e inizializza i parametri per l'aggiornamento dei pesi.

        Args:
            flag (bool): Se True, abilita l'algoritmo Rprop. Se False, lo disabilita.
            eta_minus (float, optional): Fattore di riduzione per il passo di aggiornamento quando il gradiente cambia segno. Default: 0.5.
            eta_plus (float, optional): Fattore di incremento per il passo di aggiornamento quando il gradiente mantiene il segno. Default: 1.2.
            delta_min (float, optional): Valore minimo per il passo di aggiornamento dei pesi. Default: 1e-6.
            delta_max (float, optional): Valore massimo per il passo di aggiornamento dei pesi. Default: 50.
        """
        self.rprop_active = flag
        self.eta_minus = eta_minus
        self.eta_plus = eta_plus
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.deltas_matrix_w = None
        self.deltas_weights_matrix_w = None
        self.derivatives_matrix_w = None
        self.deltas_matrix_b = None
        self.deltas_weights_matrix_b = None
        self.derivatives_matrix_b = None

    def softmax_function(self, output_data):
        """
        Calcola la funzione softmax su una matrice (num_esempi x 10).

        Args:
            output_data (np.ndarray): Matrice di input su cui applicare la funzione softmax riga per riga.

        Returns:
            np.ndarray: Matrice di input trasformata tramite softmax: i valori per riga
                        rappresentano probabilità che sommano a 1.
        """

        # Applica la funzione softmax lungo l'asse 1 (dimensione delle classi)
        return sc.softmax(output_data, axis=1)

    def forward_propagation(self, input_data):
        """
        Esegue la propagazione in avanti attraverso la rete neurale, calcolando l'output finale.

        Args:
            input_data (np.ndarray): Matrice di input di forma (num_esempi x num_feature),
                                    dove ogni riga rappresenta un esempio.

        Returns:
            np.ndarray: Output finale della rete, una matrice (num_esempi x 10). Se `self.softmax` è True,
                        l'output è trasformato con la funzione softmax per ottenere probabilità.
        """

        # Inizializza l'output corrente con i dati di input
        curr_output = np.array(input_data)

        # Propaga i dati attraverso ogni layer della rete
        for layer in self.layers:
            curr_output = layer.forward_pass(curr_output)

        # Applica la softmax all'output finale (se abilitato nel costruttore)
        if self.softmax:
            curr_output = self.softmax_function(curr_output)

        # Restituisce l'output finale della rete
        return curr_output

    def add_layer(self, n_neurons, activation_function, derivative_activation):
        """
        Aggiunge un nuovo layer alla rete neurale.

        Args:
            n_neurons (int): Numero di neuroni nel nuovo layer.
            activation_function (function): Funzione di attivazione da applicare nei neuroni del layer.
            derivative_activation (function): Derivata della funzione di attivazione (necessaria per la backpropagation).
        """
        # Determina il numero di neuroni di input per il nuovo layer
        if not self.layers:
            # Se è il primo layer, usa il numero di feature
            layer_input_size = self.input_size
        else:
            # Altrimenti, usa il numero di neuroni dell'ultimo layer
            layer_input_size = self.layers[-1].n_neurons

        # Crea e aggiunge il nuovo layer alla rete neurale
        self.layers.append(Layer(layer_input_size, n_neurons,
                           activation_function, derivative_activation))

    def backward_propagation(self, output_data, labels):
        """
        Esegue la propagazione all'indietro per aggiornare i pesi della rete neurale.

        Args:
            output_data (np.ndarray): L'output dell'ultima forward propagation (num_esempi x 10).
            labels (np.ndarray): Le etichette corrette per gli esempi in one-hot (num_esempi x 10).

        Returns:
            deltas (list): Una lista (dim: num_layer) di matrici di delta (dim: num_weights = num_input_neurons x num_neurons)
        """

        deltas = []

        # Calcola l'errore con la funzione di errore (stabilita dal costruttore)
        error = self.error_function_derivative(output_data, labels)

        # Calcola i delta del layer di output
        output_layer_deltas = error * \
            self.layers[-1].derivative_activation(
                self.layers[-1].unactived_output)

        # Aggiungi i delta del layer di output alla lista dei delta (prima vuota)
        deltas.insert(0, output_layer_deltas)

        # Calcola i delta degli altri layer (partendo dal penultimo, arrivando al primo)
        for l in range(len(self.layers) - 2, -1, -1):
            # Calcola i delta del layer interno l-esimo
            curr_delta = deltas[0].dot(self.layers[l + 1].weights.T) * \
                self.layers[l].derivative_activation(
                    self.layers[l].unactived_output)
            # Aggiungi i delta del layer interno l-esimo alla lista
            deltas.insert(0, curr_delta)

        return deltas  # Restituisci la lista dei delta

    def get_copy(self):
        """
        Crea una copia esatta della rete neurale corrente.

        Returns:
            new_net (NeuralNetwork): Una copia di se stessa
        """
        # Crea una nuova rete neurale con le stesse configurazioni della rete corrente
        new_net = NeuralNetwork(self.input_size, self.learning_rate,
                                self.error_function_name, self.softmax, self.patience)

        # Aggiungi i layer della rete corrente alla nuova rete, mantenendo le stesse configurazioni
        for l in range(len(self.layers)):
            new_net.add_layer(self.layers[l].get_n_neurons(
            ), self.layers[l].activation_function, self.layers[l].derivative_activation)

            # Copia i pesi e i bias del layer l-esimo nella nuova rete
            new_net.layers[-1].set_weights(self.layers[l].get_weights())
            new_net.layers[-1].set_bias(self.layers[l].get_bias())

        # Restituisce la nuova rete neurale con le stesse configurazioni e parametri
        return new_net

    def set_parameters_from_net(self, other_net):
        """
        Imposta i pesi e i bias della rete neurale corrente utilizzando quelli di un'altra rete.

        Args:
            other_net (NeuralNetwork): La rete neurale da cui copiare i pesi e i bias.
        """
        # Itera su tutti i layer della rete 'other_net'
        for l in range(len(other_net.layers)):
            # copia i pesi del layer l-esimo
            self.layers[l].set_weights(other_net.layers[l].get_weights())

            # copia i bias del layer l-esimo
            self.layers[l].set_bias(other_net.layers[l].get_bias())

    def default_update_rule(self, layer_index, layer_weight_derivatives, layer_bias_derivatives):
        """
        Aggiorna i pesi e i bias di un layer utilizzando la regola di aggiornamento standard.

        Args:
            layer_index (int): L'indice del layer di cui aggiornare pesi e bias.
            layer_weight_derivatives (np.ndarray): Le derivate dei pesi del layer specificato.
            layer_bias_derivatives (np.ndarray): Le derivate dei bias del layer specificato.
        """
        l = layer_index

        # Aggiorna i pesi sottraendo il prodotto tra il learning rate e le derivate dei pesi
        self.layers[l].weights -= self.learning_rate * layer_weight_derivatives

        # Aggiorna i bias del layer sottraendo il prodotto tra il learning rate e le derivate dei bias
        self.layers[l].bias -= self.learning_rate * layer_bias_derivatives

    def rprop_update_rule(self, layer_index, layer_weight_derivatives, layer_bias_derivatives):
        """
        Aggiorna i pesi e i bias di un layer utilizzando la regola di aggiornamento Rprop (Resilient Backpropagation).

        Args:
            layer_index (int): L'indice del layer di cui aggiornare pesi e bias.
            layer_weight_derivatives (np.ndarray): Le derivate correnti dei pesi del layer specificato.
            layer_bias_derivatives (np.ndarray): Le derivate correnti dei bias del layer specificato.
        """
        l = layer_index

        # Se la matrice Δ del layer corrente è None (Cioè ci troviamo alla prima epoca di update) inizializziamo adeguatamente le matrici dei valori di update
        if self.deltas_matrix_w[l] is None:
            # derivatives_matrix_w/b contiene la derivata parziale rispetto ai pesi/bias al tempo t-1, la assegnamo a 0 forzando il prodotto tra le derivate ad essere 0
            # in questo modo garantiamo il primo update dei pesi/bias sia regolare
            self.derivatives_matrix_w[l] = np.full(
                self.layers[l].weights.shape, 0)
            self.derivatives_matrix_b[l] = np.full(
                self.layers[l].bias.shape, 0)
            # Come suggerito nel paper, inizializziamo i delta al valore Δ₀ = 0.1
            self.deltas_matrix_w[l] = np.full(
                self.layers[l].weights.shape, 0.1)
            self.deltas_matrix_b[l] = np.full(self.layers[l].bias.shape, 0.1)

            # Operazioni facoltative (saranno sovrascritte in seguito), ma forzano il dtype dell'ndarray a float
            # (in mancanza di queste operazioni, Python assegnerà come dtype object, causando problemi con le funzioni matematiche di cui facciamo uso per aggiornare i pesi)
            self.deltas_weights_matrix_w[l] = np.full(
                self.layers[l].weights.shape, 0)
            self.deltas_weights_matrix_b[l] = np.full(
                self.layers[l].bias.shape, 0)

        # Calcoliamo le matrici dei prodotti tra le derivate al tempo t-1 ed al tempo t
        weight_grad_product = self.derivatives_matrix_w[l] * \
            layer_weight_derivatives
        bias_grad_product = self.derivatives_matrix_b[l] * \
            layer_bias_derivatives

        # Aggiorniamo posizionalmente le matrici Δ, in base al segno del prodotto tra le derivate, come da paper
        self.deltas_matrix_w[l] = np.where(weight_grad_product > 0,  # in caso di segno positivo
                                           # moltiplichiamo Δᵢⱼ per η⁺ (o utilizziamo Δmax)
                                           np.minimum(
                                               self.deltas_matrix_w[l] * self.eta_plus, self.delta_max),
                                           np.where(weight_grad_product < 0,  # in caso di segno negativo
                                                    # moltiplichiamo Δᵢⱼ per η⁻ (o utilizziamo Δmin)
                                                    np.maximum(
                                                        self.deltas_matrix_w[l] * self.eta_minus, self.delta_min),
                                                    self.deltas_matrix_w[l]))  # in caso di segno nullo, update regolare

        # Aggiorniamo posizionalmente le matrici Δ, in base al segno del prodotto tra le derivate, come da paper
        self.deltas_matrix_b[l] = np.where(bias_grad_product > 0,  # in caso di segno positivo
                                           # moltiplichiamo Δᵢⱼ per η⁺, aumentiamo il passo di update (o utilizziamo Δmax)
                                           np.minimum(
                                               self.deltas_matrix_b[l] * self.eta_plus, self.delta_max),
                                           np.where(bias_grad_product < 0,  # in caso di segno negativo
                                                    # moltiplichiamo Δᵢⱼ per η⁻, diminuiamo il passo di update (o utilizziamo Δmin)
                                                    np.maximum(
                                                        self.deltas_matrix_b[l] * self.eta_minus, self.delta_min),
                                                    self.deltas_matrix_b[l]))  # in caso di segno nullo, update regolare

        # Aggiorniamo posizionalmente le matrici Δw, in base al segno del prodotto tra le derivate, come da paper
        self.deltas_weights_matrix_w[l] = np.where(weight_grad_product < 0,  # in caso di segno negativo
                                                   # lasciamo invariato Δwᵢⱼ
                                                   self.deltas_weights_matrix_w[l],
                                                   -np.sign(layer_weight_derivatives) * self.deltas_matrix_w[l])  # altrimenti, aggiorniamo Δwᵢⱼ in base al segno della derivata corrente
        self.deltas_weights_matrix_b[l] = np.where(bias_grad_product < 0,  # in caso di segno negativo
                                                   # lasciamo invariato Δwᵢⱼ
                                                   self.deltas_weights_matrix_b[l],
                                                   -np.sign(layer_bias_derivatives) * self.deltas_matrix_b[l])  # altrimenti, aggiorniamo Δwᵢⱼ in base al segno della derivata corrente

        # Aggiorniamo posizionalmente i pesi ed i bias in base alla matrice Δw
        self.layers[l].weights = np.where(weight_grad_product < 0,  # in caso di segno negativo
                                          # torniamo indietro di Δwᵢⱼ
                                          self.layers[l].weights -
                                          self.deltas_weights_matrix_w[l],
                                          self.layers[l].weights + self.deltas_weights_matrix_w[l])  # altrimenti, avanziamo di Δwᵢⱼ
        self.layers[l].bias = np.where(bias_grad_product < 0,  # in caso di segno negativo
                                       # torniamo indietro di Δwᵢⱼ
                                       self.layers[l].bias -
                                       self.deltas_weights_matrix_b[l],
                                       self.layers[l].bias + self.deltas_weights_matrix_b[l])  # altrimenti, avanziamo di Δwᵢⱼ

        # Aggiorniamo le matrici delle derivate al tempo t (alla prossima iterazione queste matrici conterranno correttamente le derivate al tempo t-1)
        self.derivatives_matrix_w[l] = np.where(weight_grad_product < 0,  # se il prodotto tra le derivate ha segno negativo
                                                0,  # assegniamo 0, in modo da evitare il double punishment
                                                layer_weight_derivatives)  # altrimenti assegniamo la derivata corrente
        self.derivatives_matrix_b[l] = np.where(bias_grad_product < 0,  # se il prodotto tra le derivate ha segno negativo
                                                0,  # assegniamo 0, in modo da evitare il double punishment
                                                layer_bias_derivatives)  # altrimenti assegniamo la derivata corrente

    def train_epoch(self, dataset_train, labels_train, batch_size):
        """
        Esegue il training della rete su una epoca

        Args:
            dataset_train (np.ndarray): Il dataset di training (num_esempi x 10).
            labels_train (np.ndarray): Le etichette del dataset di training (num_esempi x 10).
            batch_size (int): La dimensione dei minibatch o del batch per il training.
        """
        # Se operiamo in minibatch, permutiamo il dataset
        if batch_size < len(dataset_train):
            # Creiamo un array di indici per lo shuffle
            indices = np.arange(len(dataset_train))
            np.random.shuffle(indices)
            dataset_train = dataset_train[indices]  # Funziona con array NumPy
            labels_train = labels_train[indices]

        else:
            # Altrimenti, operiamo in batch
            batch_size = len(dataset_train)

        # Loop su tutti i minibatch (o una singola iterazione se in batch)
        for b in range(0, len(dataset_train), batch_size):
            # Estrai i dati e le etichette per il minibatch corrente
            batch_data = dataset_train[b:b + batch_size]
            batch_labels = labels_train[b:b + batch_size]

            # Propagazione in avanti sul minibatch corrente con final_output di dim: (batch_size x 10)
            final_output = self.forward_propagation(batch_data)

            # Propagazione all'indietro per calcolare i delta
            deltas = self.backward_propagation(final_output, batch_labels)

            # Aggiorniamo i pesi e i bias per ogni layer
            for l in range(len(self.layers)):
                if (l == 0):
                    # Se siamo nel primo layer, l'input coincide col minibatch corrente
                    output_j = batch_data
                else:
                    # Altrimenti, l'input è l'output del layer precedente
                    # Output del layer precedente (z_j nei calcoli)
                    output_j = self.layers[l-1].output

                # Calcolo delle derivate dei pesi e dei bias del layer l-esimo
                layer_weight_derivatives = output_j.T.dot(
                    deltas[l])  # Derivata rispetto ai pesi
                layer_bias_derivatives = np.sum(
                    deltas[l], axis=0, keepdims=True)  # Derivata rispetto ai bias

                # Se la Rprop non è attiva, aggiorniamo i pesi con la regola standard
                if (self.rprop_active == False):
                    self.default_update_rule(
                        l, layer_weight_derivatives, layer_bias_derivatives)
                # Se la Rprop è attiva, aggiorniamo i pesi con l'algoritmo della Rprop
                else:
                    self.rprop_update_rule(
                        l, layer_weight_derivatives, layer_bias_derivatives)

    def train_network(self, dataset_train, labels_train, dataset_val, labels_val, epochs, batch_size):
        """
        Esegue il training della rete su tutte le epoche.

        Args:
            dataset_train (np.ndarray): Il dataset di addestramento.
            labels_train (np.ndarray): Le etichette del dataset di addestramento.
            dataset_val (np.ndarray): Il dataset di validazione.
            labels_val (np.ndarray): Le etichette del dataset di validazione.
            epochs (int): Numero di epoche.
            batch_size (int): La dimensione dei minibatch per l'allenamento.

        Returns:
            tuple: Il modello allenato, il miglior modello (con minore errore di validazione),
                  lista degli errori di training e la lista degli errori di validazione.
        """

        # Verifica se la softmax è usata con un solo neurone di output
        if (self.error_function == 'cross_entropy' and self.layers[-1].n_neurons == 1 and self.softmax):
            raise ValueError(
                "Utilizzare la softmax in presenza di un solo neurone di output è errato")

        # Verifica se la softmax è omessa e c'è più di un neurone di output
        if (self.error_function == 'cross_entropy' and self.layers[-1].n_neurons != 1 and not self.softmax):
            raise ValueError(
                "Non utilizzare la softmax in presenza di multipli neuroni di output è errato")

        # Verifica se si sta utilizzando la Rprop con minibatch o online, il che non è consentito
        if (self.rprop_active and batch_size < len(dataset_train)):
            raise ValueError("Non utilizzare la RPROP in minibatch o online")

        # Inizializza le matrici usate dalla Rpop, se attivata
        if (self.rprop_active == True):
            self.deltas_matrix_w = np.full(len(self.layers), None)
            self.deltas_weights_matrix_w = np.full(len(self.layers), None)
            self.derivatives_matrix_w = np.full(len(self.layers), None)
            self.deltas_matrix_b = np.full(len(self.layers), None)
            self.deltas_weights_matrix_b = np.full(len(self.layers), None)
            self.derivatives_matrix_b = np.full(len(self.layers), None)

        # Copia iniziale del modello
        best_net = self.get_copy()

        # Inizializzazione della lista degli errori di training
        errors_training_list = []

        # Calcolo dell'errore di training sulla prima rete (non allenata)
        first_train_output = self.forward_propagation(dataset_train)
        first_train_error = self.error_function(
            first_train_output, labels_train)
        errors_training_list.append(first_train_error)

        # Inizializzazione della lista degli errori di validation
        errors_validation_list = []

        # Calcolo dell'errore di validation sulla prima rete (non allenata)
        first_val_output = self.forward_propagation(dataset_val)
        first_val_error = self.error_function(first_val_output, labels_val)
        errors_validation_list.append(first_val_error)

        # Inizializzazione dell'errore di validazione minimo
        min_val_error = first_val_error
        wait = 0  # Conta le epoche senza miglioramenti

        # Stampa gli errori per la prima epoca
        print(
            f"Epoch {0}/{epochs}, Errore su training set: {errors_training_list[0]}, Errore su validation set: {errors_validation_list[0]}")

        # Loop per ogni epoca
        for epoch in range(epochs):
            # Allena il modello per l'epoca corrente
            self.train_epoch(dataset_train, labels_train, batch_size)

            # Calcola l'errore sul dataset di training per l'epoca corrente
            train_output = self.forward_propagation(dataset_train)
            train_error = self.error_function(train_output, labels_train)
            errors_training_list.append(train_error)

            # Calcola l'errore sul dataset di validation per l'epoca corrente
            val_output = self.forward_propagation(dataset_val)
            val_error = self.error_function(val_output, labels_val)
            errors_validation_list.append(val_error)

            # Se l'errore di validazione migliora, aggiorna il modello migliore
            if (val_error < min_val_error):
                best_net = self.get_copy()
                min_val_error = val_error
                wait = 0  # Reset del contatore di epoche senza miglioramenti
            else:
                wait += 1  # Altrimenti, incrementa il contatore se l'errore non migliora

            # Stampa gli errori per l'epoca corrente
            print(
                f"Epoch {epoch+1}/{epochs}, Errore su training set: {errors_training_list[epoch+1]}, Errore su validation set: {errors_validation_list[epoch+1]}")

            # Early stopping se non ci sono miglioramenti dopo 'patience' epoche
            if self.patience < wait:
                print(f"Early stopping all'epoca {epoch+1}")
                break

        return self, best_net, errors_training_list, errors_validation_list


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


def k_fold_cross_validation(X, y, k, epochs, batch_size, input_size, hidden_size, output_size, learning_rate, eta_minus=0.5, eta_plus=1.2, delta_min=1e-6, delta_max=50):
    """
    Esegue il k-fold cross-validation per allenare e testare un modello di rete neurale.

    Args:
        X (np.ndarray): Dati di input.
        y (np.ndarray): Etichette del dataset.
        k (int): Numero di fold nel k-fold cross-validation.
        epochs (int): Numero di epoche per l'allenamento.
        batch_size (int): Dimensione dei minibatch per l'allenamento.
        input_size (int): Numero di caratteristiche (dimensione input).
        hidden_size (int): Numero di neuroni nel layer nascosto.
        output_size (int): Numero di neuroni nell'output (dimensione output).
        learning_rate (float): Tasso di apprendimento.
        eta_minus (float): Parametro di Rprop per l'aggiornamento dei pesi (default: 0.5).
        eta_plus (float): Parametro di Rprop per l'aggiornamento dei pesi (default: 1.2).
        delta_min (float): Minimo valore per il delta di Rprop (default: 1e-6).
        delta_max (float): Massimo valore per il delta di Rprop (default: 50).

    Returns:
        tuple: Una tupla contenente gli errori e le accuratezze di ciascun fold.
    """

    if (k <= 2):
        raise ValueError("Il numero di fold non può essere inferiore a 3")

    # Liste per memorizzare gli errori e le accuratezze di ciascun fold
    errors = []
    accuracies = []

    # Creazione dei fold utilizzando StratifiedKFold per mantenere la distribuzione delle classi
    skf = sk.StratifiedKFold(n_splits=k, shuffle=True, random_state=53)

    # skf.split è una lista di coppie di array, dove:
    # il primo elemento è il training set dell'i-esima iterazione;
    # il secondo elemento è il test set della stessa iterazione (Sono complementari)
    splits = list(skf.split(X, y))

    # Ciclo attraverso i fold
    for i in range(k):
        # Il fold corrente è usato per testing
        test_fold = i
        # Il prossimo fold è usato per validation
        val_fold = (i + 1) % k

        # Qui accediamo al secondo elemento (cioè il test set) dell'i-esima iterazione
        test_index = splits[test_fold][1]
        # Qui accediamo al secondo elemento (cioè il test set) dell'i+1-esima iterazione
        val_index = splits[val_fold][1]

        # Visualizza l'assegnazione dei fold
        # Segna i fold per il training
        fold_visual = ["T"] * k
        # Segna il fold di test
        fold_visual[test_fold] = "E"
        # Segna il fold di validation
        fold_visual[val_fold] = "V"
        print(f"Fold {i + 1}: " + " | ".join(fold_visual))
        print("-" * 50)

        # Prendiamo gli indici del training set come complemento dell'unione dei due set creati
        train_index = np.setdiff1d(
            np.arange(len(X)), np.concatenate((test_index, val_index)))

        # Estrazione dei dati dai rispettivi set secondo gli indici scelti
        X_train = X.iloc[train_index]
        X_validation = X.iloc[val_index]
        X_test = X.iloc[test_index]
        # Estrazione delle etichette dei dati dai rispettivi set secondo gli indici scelti
        y_train = y.iloc[train_index]
        y_validation = y.iloc[val_index]
        y_test = y.iloc[test_index]

        # One-hot encoding delle etichette
        y_train_onehot = one_hot_encode(y_train)
        y_validation_onehot = one_hot_encode(y_validation)
        y_test_onehot = one_hot_encode(y_test)

        # Converti i dati in array numpy
        X_train = X_train.to_numpy()
        X_validation = X_validation.to_numpy()
        X_test = X_test.to_numpy()

        # Creazione della rete neurale
        nn = NeuralNetwork(input_size, learning_rate,
                           'cross_entropy', True, patience=30)
        # Aggiungi il layer nascosto
        nn.add_layer(hidden_size, sigmoid, der_sigmoid)
        # Aggiungi il layer di output
        nn.add_layer(output_size, identity, der_identity)

        # Configura Rprop
        nn.set_rprop(True, eta_minus, eta_plus, delta_min, delta_max)

        # Training del modello
        nn, best_net, epochs_errors_t, epochs_errors_v = nn.train_network(
            X_train, y_train_onehot, X_validation, y_validation_onehot, epochs, batch_size)

        # Visualizzazione degli errori durante le epoche
        plot_epochs_errors(epochs_errors_t, epochs_errors_v,
                           i, f"η-={eta_minus}, η+={eta_plus}, h={hidden_size}")

        # Ripristina i parametri del miglior modello
        nn.set_parameters_from_net(best_net)

        # Valutazione sul test set
        final_output_test = nn.forward_propagation(X_test)
        error = np.mean(nn.error_function(final_output_test, y_test_onehot))
        errors.append(error)

        # Calcola l'accuratezza
        predictions = np.argmax(final_output_test, axis=1)
        classes = np.argmax(y_test_onehot, axis=1)
        accuracy = np.mean(predictions == classes)
        accuracies.append(accuracy)

        # Stampa l'accuratezza e l'errore per il fold corrente
        print(f"Accuratezza su test set: {accuracy}")
        print(f"Errore nell'iterazione {i+1}: {error}")
        print("-" * 50)

    # Restituisci gli errori e le accuratezze
    return errors, accuracies


def grid_search(X, y, k, epochs, batch_size, input_size, output_size, learning_rate, hyperparameters, plot_data=True, export_csv=True):
    """
      Esegue una ricerca a griglia sugli iperparametri per trovare la combinazione ottimale di iperparametri.

      Args:
          X (np.ndarray): Dati di input.
          y (np.ndarray): Etichette del dataset.
          k (int): Numero di fold nel k-fold cross-validation.
          epochs (int): Numero di epoche per l'allenamento.
          batch_size (int): Dimensione dei minibatch per l'allenamento.
          input_size (int): Numero di caratteristiche (dimensione input).
          output_size (int): Numero di neuroni nell'output (dimensione output).
          learning_rate (float): Tasso di apprendimento.
          hyperparameters (dict): Dizionario contenente gli iperparametri da testare.
          plot_data (bool): Se True, plotta i grafici per errori e accuratezze.
          export_csv (bool): Se True, esporta i risultati in un file CSV.

      Returns:
          tuple: Una tupla contenente gli errori, le accuratezze e i parametri testati.
      """

    # Estrai gli iperparametri dal dizionario
    h_eta_minusses = hyperparameters["h_eta_minusses"]
    h_eta_plusses = hyperparameters["h_eta_plusses"]
    h_neurons = hyperparameters["h_neurons"]

    # Inizializza il miglior errore con un valore molto alto
    best_error = float('inf')
    # Memorizzerà i migliori iperparametri
    best_params = None

    all_errors = []
    all_accuracies = []
    param_labels = []

    # Nome del file CSV in cui verranno esportati i risultati
    csv_filename = "Risultati_test_grid_search.csv"

    # Ricrea il file CSV ad ogni iterazione se non esiste
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Eta minus", "Eta plus", "Numero neuroni",
                         "Errore medio", "Std errore",
                         "Accuracy media", "Std accuracy", "Tempo esecuzione"])

    # Ciclo principale che esplora tutte le combinazioni di iperparametri
    for em in h_eta_minusses:
        for ep in h_eta_plusses:
            for n in h_neurons:
                # Stampa i parametri che si stanno testando
                print(
                    f"\nTesting con eta_minus={em}, eta_plus={ep}, hidden_neurons={n}...")

                # Memorizza l'orario di inizio per calcolare il tempo di esecuzione
                start_k_fold_time = time.time()

                # Esegui il k-fold cross-validation con i parametri correnti
                epoch_errors, epoch_accuracies = k_fold_cross_validation(
                    X, y, k, epochs, batch_size, input_size, n, output_size, learning_rate,
                    eta_minus=em, eta_plus=ep
                )

                # Calcola il tempo di esecuzione per il k-fold cross-validation
                end_k_fold_time = time.time()
                elapsed_k_fold_time = round(
                    end_k_fold_time - start_k_fold_time, 4)

                # Stampa il tempo impiegato
                print(f"Elapsed time: {elapsed_k_fold_time:.2f} seconds")

                # calcola la media e la deviazione standard delle accuracy
                mean_error = round(np.mean(epoch_errors), 4)
                std_error = round(np.std(epoch_errors), 4)

                # Calcola la media e la deviazione standard degli errori
                mean_accuracy = round(np.mean(epoch_accuracies), 4)
                std_accuracy = round(np.std(epoch_accuracies), 4)

                # Stampa i risultati per la combinazione corrente di iperparametri
                print(
                    f"Errore medio: {mean_error:.4f}, Accuracy media: {mean_accuracy:.4f}")

                # Aggiungi gli errori, le accuratezze e i parametri testati alle rispettive liste
                all_errors.append(epoch_errors)
                all_accuracies.append(epoch_accuracies)
                param_labels.append({
                    "eta_minus": em,
                    "eta_plus": ep,
                    "hidden_neurons": n
                })

                # Se plot_data è True, disegna i grafici per errori e accuratezze
                if (plot_data):
                    plot_fold_errors(epoch_errors, param_labels)
                    plot_fold_accuracies(epoch_accuracies, param_labels)

                # Se export_csv è True, esporta i risultati nel file CSV
                if (export_csv):
                    with open(csv_filename, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([em, ep, n,
                                        mean_error, std_error,
                                        mean_accuracy, std_accuracy, elapsed_k_fold_time])

    return all_errors, all_accuracies, param_labels


def random_search(X, y, k, epochs, batch_size, input_size, output_size, learning_rate, hyperparameters, n_iter=5, plot_data=True, export_csv=True):
    """
    Esegue una ricerca random sui range di iperparametri per trovare la combinazione ottimale.

    Args:
        X (array-like): Dati di input.
        y (array-like): Etichette target.
        k (int): Numero di fold per la cross-validation.
        epochs (int): Numero di epoche di addestramento.
        batch_size (int): Dimensione del batch per l'addestramento.
        input_size (int): Numero di feature in input.
        output_size (int): Numero di classi in output.
        learning_rate (float): Tasso di apprendimento.
        hyperparameters (dict): Dizionario contenente i range degli iperparametri da ottimizzare.
        n_iter (int, opzionale): Numero di combinazioni casuali da testare. Default = 5.
        plot_data (bool, opzionale): Se True, genera i grafici degli errori e delle accuratezze. Default = True.
        export_csv (bool, opzionale): Se True, esporta i risultati in un file CSV. Default = True.

    Returns:
        tuple: Contiene tre elementi:
            - all_errors (list): Lista contenente gli errori medi per ogni combinazione testata.
            - all_accuracies (list): Lista contenente le accuratezze medie per ogni combinazione testata.
            - param_labels (list): Lista dei parametri testati con i rispettivi valori.
    """
    h_eta_minusses = hyperparameters["h_eta_minusses"]
    h_eta_plusses = hyperparameters["h_eta_plusses"]
    neuron_range = hyperparameters["h_neurons"]

    # Inizializza il miglior errore con un valore molto alto
    best_error = float('inf')
    # Memorizzerà i migliori iperparametri
    best_params = None

    all_errors = []
    all_accuracies = []
    param_labels = []

    # Nome del file CSV
    csv_filename = "Risultati_test_random_search.csv"

    # Ricrea il file CSV ad ogni iterazione se non esiste
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Eta minus", "Eta plus", "Numero neuroni",
                         "Errore medio", "Std errore",
                         "Accuracy media", "Std accuracy", "Tempo esecuzione"])

    for _ in range(n_iter):
        # Seleziona casualmente i valori dagli iperparametri
        eta_minus = round(random.uniform(
            h_eta_minusses[0], h_eta_minusses[1]), 4)
        eta_plus = round(random.uniform(h_eta_plusses[0], h_eta_plusses[1]), 4)
        hidden_neurons = random.randint(neuron_range[0], neuron_range[1])

        print(
            f"\nTesting con eta_minus={eta_minus}, eta_plus={eta_plus}, hidden_neurons={hidden_neurons}...")

        # Memorizza l'orario di inizio per calcolare il tempo di esecuzione
        start_k_fold_time = time.time()

        # Esegui il k-fold cross-validation con i parametri correnti
        epoch_errors, epoch_accuracies = k_fold_cross_validation(
            X, y, k, epochs, batch_size, input_size, hidden_neurons, output_size, learning_rate,
            eta_minus=eta_minus, eta_plus=eta_plus
        )

        # Calcola il tempo di esecuzione per il k-fold cross-validation
        end_k_fold_time = time.time()
        elapsed_k_fold_time = round(end_k_fold_time - start_k_fold_time, 4)

        # Stampa il tempo impiegato
        print(f"Elapsed time: {elapsed_k_fold_time:.2f} seconds")

        # calcola la media e la deviazione standard delle accuracy
        mean_error = round(np.mean(epoch_errors), 4)
        std_error = round(np.std(epoch_errors), 4)

        # Calcola la media e la deviazione standard degli errori
        mean_accuracy = round(np.mean(epoch_accuracies), 4)
        std_accuracy = round(np.std(epoch_accuracies), 4)

        # Stampa i risultati per la combinazione corrente di iperparametri
        print(
            f"Errore medio: {mean_error:.4f}, Accuracy media: {mean_accuracy:.4f}")

        # Aggiungi gli errori, le accuratezze e i parametri testati alle rispettive liste
        all_errors.append(epoch_errors)
        all_accuracies.append(epoch_accuracies)
        param_labels.append({
            "eta_minus": eta_minus,
            "eta_plus": eta_plus,
            "hidden_neurons": hidden_neurons
        })

        # Se plot_data è True, disegna i grafici per errori e accuratezze
        if (plot_data):
            plot_fold_errors(epoch_errors, param_labels)
            plot_fold_accuracies(epoch_accuracies, param_labels)

        # Se export_csv è True, esporta i risultati nel file CSV
        if (export_csv):
            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([eta_minus, eta_plus, hidden_neurons,
                                mean_error, std_error,
                                mean_accuracy, std_accuracy, elapsed_k_fold_time])

    return all_errors, all_accuracies, param_labels
