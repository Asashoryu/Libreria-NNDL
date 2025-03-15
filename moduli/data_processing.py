from PIL import Image
from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np


def one_hot_encode(y):
    """
    Funzione che converte un vettore di n valori (tra 0 e 9) in una matrice in forma one hot, di forma (n x 10).

    Args:
        y (np.ndarray): Vettore di valori da convertire.

    Returns:
        one_hot_y: matrice di valori in forma one_hot
    """

    # Trasformiamo il vettore in input in un vettore colonna
    # (nella reshape -1 indica di adeguare la dimensione, quindi il vettore risultante sarà n x 1)
    y_reshaped = y.values.reshape(-1, 1)

    # Creiamo una matrice di zeri di forma (n x 10)
    one_hot_y = np.zeros((y_reshaped.shape[0], 10))

    # Preso l'i-esimo tra gli n valori originali
    for i in range(y_reshaped.shape[0]):
        # In posizione (i, valore di i) pongo un 1, ottenendo così un vettore che rappresenta l'i-esimo valore in forma one-hot
        one_hot_y[i, y_reshaped[i]] = 1

    return one_hot_y


def resize_mnist(mnist, width, height):
    """
    Funzione che prende il DataFrame del MNIST ed effettua una resize di ogni immagine.

    Args:
        mnist (pd.DataFrame): DataFrame del MNIST, con 'data' contenente le immagini sottoforma di DataFrame a sua volta.
        width (int): larghezza desiderata delle immagini
        height (int): altezza desiderata delle immagini

    Returns:
        resized_mnist (pd.DataFrame): Dataframe contenente le immagini ridimensionate
    """

    # Creiamo una lista per contenere le immagini ridimensionate
    resized_mnist_list = []
    # Per ognuna delle n immagini (mnist.data ha forma n x 784)
    for i in range(len(mnist.data)):
        # Convertiamo ogni immagine (array 1x784) in una matrice di 28x28 pixel
        image_data = mnist.data.values[i].reshape(28, 28)

        # Convertiamo in PIL Image per permettere l'operazione di resize
        pil_image = Image.fromarray(np.uint8(image_data))

        # Ridimensioniamo l'immagine in width x height pixel
        resized_image = pil_image.resize((width, height))

        # Conserviamo questa immagine ridimensionata (convertita in un np.array schiacciato in 1x(width*height)) all'interno di una lista
        resized_mnist_list.append(np.array(resized_image).flatten())

    # Convertiamo la lista di n vettori in una matrice n x (width*height)
    resized_mnist_array = np.array(resized_mnist_list)

    # Otteniamo un pandas DataFrame dall'array
    resized_mnist = pd.DataFrame(resized_mnist_array)

    return resized_mnist


def load_mnist_784(dataset_size=70000, width_size=28, height_size=28, normalize_X=True):
    """
    Carica il numero richiesto di esempi dal MNIST 784 (28 x 28) e li ridimensiona secondo le dimensioni specificate.

    Args:
        dataset_size (int): Numero massimo di campioni da caricare. Default 70'000 (dimensione completa di MNIST).
        width_size (int): Larghezza desiderata delle immagini dopo il ridimensionamento. Default 28 (dimensione originale).
        height_size (int): Altezza desiderata delle immagini dopo il ridimensionamento. Default 28 (dimensione originale).
        normalize_X (bool): Se True, normalizza i valori dei pixel nel range [0,1] dividendo per 255. Default True.

    Returns:
        tuple:
            - X_df (pd.DataFrame): DataFrame contenente i dati delle immagini (ogni riga rappresenta un'immagine).
            - y_series (pd.Series): Serie contenente le etichette corrispondenti alle immagini.
    """

    # Carica il dataset MNIST (70.000 immagini, 784 pixel per immagine)
    mnist = fetch_openml('mnist_784')

    # Estrai le immagini e le etichette dal dataset
    X, y = mnist['data'], mnist['target'].astype(int)

    # Ridimensiona le immagini alla larghezza e altezza desiderate
    resized_mnist = resize_mnist(mnist, width_size, height_size)
    X, y = resized_mnist, mnist.target.astype(int)

    # Se richiesto, normalizza i valori dei pixel nel range [0,1]
    if normalize_X:
        X = X / 255.0

    # Seleziona il numero desiderato di campioni dal dataset
    X_subset = X[:dataset_size]
    y_subset = y[:dataset_size]

    # Converte i dati delle immagini in un DataFrame di Pandas
    X_df = pd.DataFrame(X_subset)

    # Converte le etichette in una Serie di Pandas
    y_series = pd.Series(y_subset, name="label")

    return X_df, y_series
