import matplotlib.pyplot as plt

def plot_list(data: list[float], text1: str, text2: str, headline: str):
    """Plots a list of doubles with custom axis labels and a title."""
    plt.plot(data)
    plt.ylabel(text1)
    plt.xlabel(text2)
    plt.title(headline)
    plt.show()


    