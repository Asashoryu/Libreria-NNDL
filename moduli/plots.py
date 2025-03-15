import matplotlib.pyplot as plt


def plot_fold_errors(fold_errors, param_labels):
    """
    Disegna un grafico a barre degli errori per ogni fold.

    Args:
        fold_errors (list): Lista degli errori per ogni fold.
        param_labels (list): Lista di terne di iperparametri (eta_min, eta_max, hidden_neurons).

    """

    # Imposta il grafico
    plt.figure(figsize=(10, 6))

    # Disegna l'errore di ciascun fold (un singolo punto per ogni fold)
    bars = plt.bar(range(len(fold_errors)), fold_errors,
                   label='Errore per Fold', color='#f08080')

    # Aggiunge i valori testuali sopra le barre
    for bar in bars:
        height = bar.get_height()
        # Mostra il valore (errore) sopra la barra
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.4f}',
                 ha='center', va='bottom', fontsize=10, color='black')

    # Imposta i tick discreti sull'asse x (il numero del fold inizia da 1, quindi si adatta di conseguenza)
    plt.xticks(range(len(fold_errors)), [
               f"Fold {i+1}" for i in range(len(fold_errors))])

    # Etichette e titolo
    plt.xlabel('Fold')
    plt.ylabel('Errore')
    plt.title('Errori per ogni fold')

    # Mostra la legenda e la griglia
    plt.legend()
    plt.grid(True, axis='y')

    # Mostra il grafico
    plt.tight_layout()

    # Salva il grafico su file
    # param_labels contiene tutte le combinazioni di parametri utilizzate finora, quindi in ultima posizione avremo i parametri correnti
    param_str = "_".join(
        [f"{key}={value}" for key, value in param_labels[-1].items()])
    plt.savefig(f'plot_fold_errors_{param_str}.png')


def plot_fold_accuracies(fold_accuracies, param_labels):
    """
    Disegna un grafico a barre delle accuracy per ogni fold.

    Args:
        fold_accuracies (list): Lista delle accuracy per ogni fold.
        param_labels (list): Lista di terne di iperparametri (eta_min, eta_max, hidden_neurons).

    """
    # Imposta il grafico
    plt.figure(figsize=(10, 6))

    # Disegna l'accuratezza di ciascun fold (un singolo punto per ogni fold)
    bars = plt.bar(range(len(fold_accuracies)), fold_accuracies,
                   label='Accuratezza per Fold', color='#66cdaa')

    # Aggiunge i valori testuali sopra le barre
    for bar in bars:
        height = bar.get_height()
        # Mostra il valore (errore) sopra la barra
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.4f}',
                 ha='center', va='bottom', fontsize=10, color='black')

    # Imposta i tick discreti sull'asse x (il numero del fold inizia da 1, quindi si adatta di conseguenza)
    plt.xticks(range(len(fold_accuracies)), [
               f"Fold {i+1}" for i in range(len(fold_accuracies))])

    # Imposta l'asse Y con un intervallo da 0 a 1
    plt.ylim(0.0, 1.0)

    # Etichette e titolo
    plt.xlabel('Fold')
    plt.ylabel('Accuratezza')
    plt.title('Accuratezza per ogni fold')

    # Mostra la legenda e la griglia
    plt.legend()
    plt.grid(True, axis='y')

    # Mostra il grafico
    plt.tight_layout()

    # Salva il grafico su file
    # param_labels contiene tutte le combinazioni di parametri utilizzate finora, quindi in ultima posizione avremo i parametri correnti
    param_str = "_".join(
        [f"{key}={value}" for key, value in param_labels[-1].items()])
    plt.savefig(f'plot_fold_accuracies_{param_str}.png')


def plot_epochs_errors(epochs_errors_t, epochs_errors_v, fold_i, curr_param_labels):
    """
    Genera e salva un grafico degli errori per epoca durante il training e la validation.

    Args:
        epochs_errors_t (list): Lista degli errori sul training set per ogni epoca.
        epochs_errors_v (list): Lista degli errori sul validation set per ogni epoca.
        fold_i (int): Indice del fold corrente.
        curr_param_labels (str): Stringa contenente i parametri correnti per etichettare il file di output.
    """

    # Imposta la dimensione del grafico
    plt.figure(figsize=(10, 6))

    # Traccia l'errore sul training set in blu con linea continua
    plt.plot(range(len(epochs_errors_t)), epochs_errors_t,
             linestyle='-', color='b', label='Errore su training set')

    # Traccia l'errore sul validation set in rosso con linea tratteggiata
    plt.plot(range(len(epochs_errors_v)), epochs_errors_v,
             linestyle='--', color='r', label='Errore su validation set')

    # Trova l'epoca con l'errore minimo sul validation set
    min_val_error_epoch = epochs_errors_v.index(
        min(epochs_errors_v))  # Indice dell'epoca con errore minimo
    # Valore dell'errore minimo
    min_val_error = epochs_errors_v[min_val_error_epoch]

    # Evidenzia l'epoca con errore minimo con un punto rosso
    plt.scatter(min_val_error_epoch, min_val_error, color='r', zorder=5,
                label=f'Errore minimo {min_val_error:.4f} (Epoca {min_val_error_epoch})')

    # Imposta i tick dell'asse x con step di 10 epoche
    plt.xticks(range(0, len(epochs_errors_t), 10),
               [f"{i}" for i in range(0, len(epochs_errors_t), 10)])

    # Etichette degli assi e titolo del grafico
    plt.xlabel('Epoca')
    plt.ylabel('Errore')
    plt.title('Errore per epoca - Fold ' + str(fold_i))

    # Mostra la legenda e abilita la griglia sull'asse y
    plt.legend()
    plt.grid(True, axis='y')

    # Ottimizza la disposizione degli elementi nel grafico
    plt.tight_layout()

    # Salva il grafico come file PNG con un nome che include i parametri e il fold corrente
    plt.savefig(f'plot_epoch_errors_{curr_param_labels}_fold_{fold_i}.png')
