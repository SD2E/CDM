from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import Concatenate
from keras.optimizers import Adam
from keras import backend as K
import pandas as pd
from harness.th_model_classes.class_keras_regression import KerasRegression
import itertools


class DE_multioutput_DEA(KerasRegression):
    def __init__(self, model, model_author, model_description, epochs=25, batch_size=1000, verbose=0):
        super(DE_multioutput_DEA, self).__init__(model, model_author, model_description)
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def _fit(self, X, y):
        # checkpoint_filepath = 'sequence_only_cnn_{}.best.hdf5'.format(str(randint(1000000000, 9999999999)))
        # checkpoint_callback = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', save_best_only=True)
        # stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=3)
        # callbacks_list = [checkpoint_callback, stopping_callback]


        # Configure the output
        y_temp = pd.DataFrame(y.tolist(), index=y.index, columns=['impacted_col', 'regulation_col','logFC_col'])
        y1 = y_temp['impacted_col']
        y2 = y_temp['regulation_col']
        y3 = y_temp['logFC_col']

        print("Length of ys", len(y1), len(y2))
        self.model.fit(X, [y1, y2,y3], epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        # self.model.load_weights(checkpoint_filepath)
        # os.remove(checkpoint_filepath)

    def _predict(self, X):

        y1, y2,y3 = self.model.predict(X)

        merged1 = list(itertools.chain(*y1))
        merged2 = list(itertools.chain(*y2))
        merged3 = list(itertools.chain(*y3))

        preds = list(zip(merged1, merged2,merged3))

        return preds


def DE_multioutput_DEA_model(num_condition_cols, emb_dim=32, batch_size=10000, epochs=25,
                                             learning_rate=1e-3):
    # Gene 1 embedding layer
    input = Input(shape=(num_condition_cols+emb_dim,))
    model = Dense(16, activation='relu')(input)
    model = Dense(4,  activation='relu')(model)

    impact_pred = Dense(1,activation='sigmoid',name='impacted_pred')(model)
    regulation_pred = Dense(1, activation='sigmoid',name='regulation_pred')(model)
    de_prediction_layer = Dense(1, activation='linear', name='de_pred')(model)


    # Create model
    model = Model(inputs=[input], outputs=[impact_pred,regulation_pred,de_prediction_layer])

    model.compile(loss={'impacted_pred': 'binary_crossentropy',
                        'regulation_pred':'binary_crossentropy',
                        'de_pred': 'mean_absolute_error'}, optimizer=Adam(lr=learning_rate),
                  metrics={'impacted_pred': 'binary_crossentropy',
                           'regulation_pred': 'binary_crossentropy',
                           'de_pred': 'mse'})

    th_model = DE_multioutput_DEA(model=model, model_author="Mohammed",
                                               model_description='DEA multioutput model', batch_size=32, epochs=50)

    return th_model
