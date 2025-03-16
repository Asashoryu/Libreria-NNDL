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

from NeuralNetwork import NeuralNetwork


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
