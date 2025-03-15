import numpy as np


def sigmoid(x):
    """
    Applica la sigmoide a un layer di neuroni.

    Args:
        x (np.ndarray): L'input è una matrice (num_esempi x num_neurons_layer).

    Returns:
        np.ndarray: Restituisce la matrice di input con la sigmoide applicata.
    """
    # Limita i valori di x per evitare overflow numerico durante il calcolo della funzione esponenziale
    x_clipped = np.clip(x, -500, 500)

    # Calcola la funzione sigmoide
    return 1 / (1 + np.exp(-x_clipped))


def der_sigmoid(x):
    """
    Calcola la derivata della funzione sigmoide.

    Args:
        x (np.ndarray): L'input è una matrice (num_esempi x num_neurons_layer).

    Returns:
        np.ndarray: Restituisce l'output della derivata della funzione sigmoide.
    """
    # Calcola la sigmoide di x
    sigm = sigmoid(x)

    # Calcola la derivata di x
    return sigm * (1 - sigm)


def identity(x):
    """
      restituisce il valore di input.

      Args:
          x (np.ndarray): L'input è una matrice (num_esempi x num_neurons_layer).

      Returns:
          np.ndarray: Restituisce lo stesso valore passato come input, senza alcuna trasformazione.
    """
    return x


def der_identity(x):
    """
      Derivata della funzione identità: restituisce sempre 1.

      Args:
          x (np.ndarray): L'input è una matrice (num_esempi x num_neurons_layer).

      Returns:
          np.ndarray: matrice di 1.
    """
    return np.ones_like(x)


def tanh(x):
    """
      unzione di attivazione tanh, che restituisce valori compresi tra -1 e 1.

      Args:
          x (np.ndarray): L'input è una matrice (num_esempi x num_neurons_layer).

      Returns:
          np.ndarrFay: Il risultato della funzione tanh applicato ad ogni elemento di x.
    """
    return np.tanh(x)


def der_tanh(x):
    """
      Derivata della funzione di attivazione tanh.

      Args:
          x (np.ndarray): L'input è una matrice (num_esempi x num_neurons_layer).

      Returns:
          np.ndarray: La derivata della funzione tanh applicata ad ogni elemento di x.
    """
    return 1 - np.tanh(x) ** 2


def relu(x):
    """
      Funzione di attivazione ReLU, restituisce:
      - 0 per tutti i valori negativi o nulli di x
      - x per tutti i valori positivi di x.

      Args:
          x (np.ndarray): L'input è una matrice (num_esempi x num_neurons_layer).

      Returns:
          np.ndarray: La funzione ReLU applicata ad ogni elemento di x.
    """
    return np.maximum(0, x)


def der_relu(x):
    """
      Derivata della funzione di attivazione ReLU, restituisce:
      - 0 per tutti i valori negativi o nulli di x.
      - 1 per tutti i valori positivi di x.

      Args:
          x (np.ndarray): L'input è una matrice (num_esempi x num_neurons_layer).

      Returns:
          np.ndarray: La derivata di ReLU applicata ad ogni elemento di x.
    """
    return np.where(x > 0, 1, 0)
