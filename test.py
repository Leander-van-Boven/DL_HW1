from train_test import get_train_test
from tensorflow.keras.applications.efficientnet\
    import EfficientNetB4, preprocess_input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from sys import exit
import tensorflow as tf
from keras.utils.np_utils import to_categorical

from set_session import initialize_session


initialize_session()


# X_train, X_test, y_train, y_test = get_train_test(
#     './data', .1, preprocess_input)

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

print('X_train', X_train.shape)
print('Y_train', y_train.shape)
# exit()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


base_efn = EfficientNetB4(
    include_top=False,
    weights=None,
    input_tensor=None,
    input_shape=(32, 32, 3),
    pooling='max'
    # classes=1070,    # 5 characters * 10 numeric * 26 roman lowercase
    # classifier_activation="softmax"
)
d_layer = Dense(10, activation='softmax')(base_efn.output)
efn = Model(inputs=base_efn.input, outputs=d_layer)
print('OUT:', efn.output.shape)

efn.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy', 'mse']
)

efn.fit(
    X_train, y_train,
    batch_size=32,
    epochs=10,
    verbose=1,
    shuffle=True,
    validation_split=.1,
    workers=0,
    use_multiprocessing=False
)

score = efn.evaluate(
    X_test, y_test,
    verbose=1,
    workers=0,
    use_multiprocessing=False
)

print('score:', score)
