from sklearn.model_selection import cross_val_score
from tensorflow.keras.applications.efficientnet import EfficientNetB4

efn = EfficientNetB4(
    include_top=False,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax"
)

results = cross_vall_score(efn, )