import numpy as np
import scipy.special as sc


from funzioni_errore import cross_entropy_with_softmax, cross_entropy_der_with_softmax, cross_entropy_binary, cross_entropy_binary_der, sum_of_squares, sum_of_squares_der
from plots import plot_fold_errors, plot_fold_accuracies, plot_epochs_errors
from data_processing import one_hot_encode, resize_mnist, load_mnist_784
from Layer import Layer


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
