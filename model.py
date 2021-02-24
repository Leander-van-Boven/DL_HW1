from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping


def run_model(architecture, batch, optimizer,
              X_train, y_train, X_test, y_test, metrics, log_dir):
    base_model = architecture(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=(32, 32, 3),
        pooling='max'
    )

    last_layer = Dense(10, activation='softmax')(base_model.output)
    model = Model(inputs=base_model.input, outputs=last_layer)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=metrics
    )
    
    tb = TensorBoard(log_dir=log_dir, histogram_freq=1)
    es = EarlyStopping(monitor='val_loss', patience=3)
    model.fit(
        X_train,
        y_train,
        batch_size=batch,
        epochs=100,
        verbose=0,
        shuffle=True,
        validation_split=.1,
        callbacks=[tb, es]
    )

    return model.evaluate(
        X_test,
        y_test,
        verbose=0
    )
