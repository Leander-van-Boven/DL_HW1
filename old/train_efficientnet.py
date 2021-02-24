from sklearn.model_selection import train_test_split
import numpy as np
from os import listdir
from sys import exit
from sklearn.model_selection import cross_val_score
from tensorflow.keras.applications.efficientnet import EfficientNetB4
from tensorflow.keras.preprocessing import image_dataset_from_directory

classes = [f.replace('.png', '') for f in listdir('./data')]
# print(classes)

ds = image_dataset_from_directory(
    './prep',
    labels='inferred',
    label_mode='categorical',
    class_names=None,
    color_mode='grayscale',
    batch_size=1,  # default 32
    image_size=(200, 50),
    # shuffle=True,
    validation_split=None)
    # returns ((None, 200, 50, 1), (None, 1070))

ds2 = list(ds)
# X_train, X_test, y_train, y_test = train_test_split(ds2, test_size=.1)
train, test = train_test_split(ds2, test_size=.1)
# print(len(train), len(test))
# exit()

# for (X, y) in ds:
#     print('X', len(X))
#     print('y', len(y))

# exit()

efn = EfficientNetB4(
    include_top=False,
    weights=None,
    input_tensor=None,
    input_shape=(200, 50, 1),
    pooling=None,
    classes=180,    # 5 characters * 10 numeric * 26 roman lowercase
    classifier_activation="softmax"
)

efn.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy', 'mse']
)

efn.fit(
    train,
    batch_size=None,  # already embedded in ds
    epochs=10,
    verbose=1,
    shuffle=True
)

score = efn.evaluate(
    test,
    verbose=1
)

print('score:', score)

# results = cross_val_score(
#     efn,
#     cv=1)