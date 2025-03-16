from data_processing import one_hot_encode, resize_mnist, load_mnist_784
from training import grid_search, random_search


def main():
    # Caricamento del dataset MNIST con 70'000 esempi e ridimensionato a 14x14 pixel per esempio
    X, y = load_mnist_784(width_size=14, height_size=14)

    # Dimensione dell'input (numero di feature per esempio)
    input_size = X.shape[1]
    # Numero di neuroni nel livello nascosto
    hidden_size = 128
    # Numero di classi (cifre da 0 a 9)
    output_size = 10
    # Tasso di apprendimento
    learning_rate = 0.0001
    # Numero di epoche per fold
    epochs = 300
    # Numero di esempi dopo i quali avviene un aggiornamento della rete neurale
    batch_size = X.shape[0]
    # Numero di fold per la cross-validation
    k = 10

    # Definizione dei valori degli iperparametri da testare nella grid search
    hyperparameters = {
        "h_eta_minusses": [0.5, 0.6, 0.7],
        "h_eta_plusses": [1.2, 1.3, 1.5],
        "h_neurons": [64, 128, 256]
    }

    # Definizione degli intervalli di ricerca per la random search (h_neurons è un intervallo di valori discreti)
    hyperparameters_ranges = {
        "h_eta_minusses": [0.4, 0.9],
        "h_eta_plusses": [1.1, 1.6],
        "h_neurons": [64, 256]
    }

    # Esecuzione della grid search per la ricerca degli iperparametri
    all_errors, all_accuracies, param_labels = grid_search(
        X, y, k, epochs, batch_size, input_size, output_size, learning_rate, hyperparameters)

    # Esecuzione della random search per la ricerca degli iperparametri
    fold_errors, fold_accuracies, param_labels = random_search(
        X, y, k, epochs, batch_size, input_size, output_size, learning_rate, hyperparameters_ranges, n_iter=10)


if __name__ == "__main__":
    main()
