from itertools import product
import os

from keras.utils.np_utils import to_categorical
from tensorflow.keras.optimizers.schedules import InverseTimeDecay
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.applications.efficientnet\
    import EfficientNetB4, EfficientNetB3

from model import run_model
from set_session import initialize_session


if __name__ == '__main__':

    #initialize_session()

    metrics = ['accuracy', 'mse']

    network = {'efnb4': EfficientNetB4, 'efnb3': EfficientNetB3}
    batch = {'16': 16, '32': 32, '64': 64}

    # Init optmizers
    lr_schedule = InverseTimeDecay(
        initial_learning_rate=0.256,
        decay_steps=2.4,
        decay_rate=0.97
    )
    optimizer = {
        'RMSprop': RMSprop(lr_schedule, rho=0.9, momentum=0.9),
        'Adam': Adam(lr_schedule)
    }

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    csv = []
    csv.append(['network', 'batch', 'optimizer'] + metrics)

    params = (X_train, y_train, X_test, y_test, metrics)

    for run in product(network, batch, optimizer):
        rstr = "network=%sbatch=%soptimizer=%s" % run
        print(rstr)
        os.makedirs('./%s' % rstr, exist_ok=True)
        res = run_model(
            network[run[0]], batch[run[1]], optimizer[run[2]], *params,
            './%s' % rstr
        )
        csv.append(list(run) + res)
        print('\t%s' % ', '.join(['%s=%s' % val for val in zip(metrics, res)]))

    csv = '\n'.join([','.join([str(i) for i in row]) for row in csv])

    with open('out.csv', 'w') as file:
        file.write(csv)
