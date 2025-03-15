import numpy as np

def cross_entropy_with_softmax(output, labels):
    """
      Calcola la funzione di errore Cross-Entropy con Softmax.

      Args:
          output (np.ndarray): La matrice di output della rete neurale (num_esempi x 10), con la softmax applicata riga per riga
          labels (np.ndarray): Le etichette corrette per il dataset in rappresentazione one-hot (num_esempi x 10)

      Returns:
          float: la somma delle Cross-Entropy con Softmax applicate a tutti gli esempi.
    """
    epsilon = 1e-12  # Piccola costante per evitare valori log(0)
    # Evita log(0) applicando un minimo epsilon agli output
    clipped_output = np.clip(output, epsilon, 1 - epsilon)
    return -np.sum(labels * np.log(clipped_output))  # Ritorna la Cross-Entropy


def cross_entropy_der_with_softmax(output, labels):
    """
      Calcola la derivata della funzione di errore Cross-Entropy con Softmax.

      Args:
          output (np.ndarray): La matrice di output della rete neurale (num_esempi x 10), con la softmax applicata riga per riga.
          labels (np.ndarray): Le etichette corrette per il dataset in rappresentazione one-hot (num_esempi x 10).

      Returns:
          np.ndarray: matrice (num_esempi x 10) che contiene le derivate della matrice output
    """
    return output - labels  # Derivate della Cross-Entropy con Softmax


def cross_entropy_binary(output, labels):
    """
      Calcola la funzione di errore Cross-Entropy binaria (1 solo neurone di output).

      Args:
          output (np.ndarray): Il vettore colonna di output della rete neurale (num_esempi), con la softmax applicata riga per riga.
          labels (np.ndarray): Il vettore delle etichette corrette per il dataset (1 o 0 per esempio) in rappresentazione one-hot (num_esempi).

      Returns:
          float: la somma delle Cross-Entropy binarie applicate a tutti gli esempi.
    """
    return (-1) * np.sum(labels * np.log(output) + (1 - labels) * np.log(1 - output))


def cross_entropy_binary_der(output, labels):
    """
      Calcola la derivata della funzione di errore Cross-Entropy binaria(1 solo neurone di output).

      Args:
          output (np.ndarray): Il vettore colonna di output della rete neurale (num_esempi), con la softmax applicata riga per riga.
          labels (np.ndarray): Il vettore delle etichette corrette per il dataset (1 o 0 per esempio) in rappresentazione one-hot (num_esempi).

      Returns:
          np.ndarray: Un array della stessa forma di 'output', contenente le derivate della funzione di errore.
    """
    return (output - labels) / (output * (1 - output))


def sum_of_squares(output, labels):
    """
    Calcola la funzione di errore della somma dei quadrati.

    Args:
        output (np.ndarray): Matrice di output (num_esempi x 10).
        labels (np.ndarray): Matrice delle etichette corrette (num_esempi x 10).

    Returns:
        float: Il valore dell'errore calcolato come la somma dei quadrati delle differenze tra
               le previsioni e le etichette reali per tutti gli esempi e tutte le classi.
    """
    # Calcola la somma dei quadrati delle differenze per ogni esempio su tutte le 10 classi.
    # La somma totale degli errori quadrati viene restituita come un valore scalare.
    return 1/2 * np.sum((output - labels) ** 2)


def sum_of_squares_der(output, labels):
    """
      Calcola la derivata della funzione di errore della somma dei quadrati.

      Args:
          output (np.ndarray): Matrice di output (num_esempi x 10).
          labels (np.ndarray): Matrice delle etichette corrette (num_esempi x 10).

      Returns:
          np.ndarray: matrice (num_esempi x 10), che rappresenta la differenza tra le previsioni e le etichette corrette.
    """
    # La derivata della somma dei quadrati per ogni esempio è la differenza tra l'output e le etichette reali
    return output - labels
