from matplotlib import pyplot as plt
from tbparse import SummaryReader

from callbacks import EarlyStopper

logdir = 'tb_accuracy_test'

if __name__ == '__main__':

    data = SummaryReader(logdir, pivot=True).scalars
    data['val/metrics'].plot()

    stopper = EarlyStopper(patience=10, min_delta=0.01)

    for i, value in enumerate(data['val/metrics']):
        if stopper.early_stop(value) and i > 50:
            print(f'Stopped at epoch {i}')
            plt.axvline(i, color='r', linestyle='--')
            break
    else:
        print('No early stopping')

    plt.show()
