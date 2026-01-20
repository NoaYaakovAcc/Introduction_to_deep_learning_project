import matplotlib
# This MUST happen before importing pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def plot_list(data: list[float], text1: str, text2: str, headline: str, save_dir: str = None):
    """Plots a list of doubles with custom axis labels and a title."""
    filename = headline.lower().replace(' ', '_').replace('over', '').strip('_') + '.png'
    plt.plot(data)
    plt.ylabel(text1)
    plt.xlabel(text2)
    plt.title(headline)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        full_path = os.path.join(save_dir, filename)
        plt.savefig(full_path)
        plt.close()
        return full_path
    else:
        plt.close()
        return None
    