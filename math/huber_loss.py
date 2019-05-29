import matplotlib.pyplot as plt
import numpy as np


def huber_loss(r, delta):
    """Huber loss function, refer to wiki https://en.wikipedia.org/wiki/Huber_loss"""
    return (abs(r) <= delta) * r ** 2 / 2 + (abs(r) > delta) * delta * (abs(r) - delta / 2)


if __name__ == '__main__':
    fig = plt.figure('Huber Loss Function', figsize=(8, 6))
    x = np.arange(-10, 10, 0.01)
    plt.plot(x, x ** 2 / 2, 'g', label='Squared Loss')
    for d in (5, 3, 1, 0.5):
        plt.plot(x, huber_loss(x, d), label=f'Huber Loss $\delta$ = {d}')

    plt.grid(True)
    plt.xlabel('Residual')
    plt.ylabel('Loss')
    plt.title('Huber Loss')
    plt.legend()

    plt.show(block=True)
