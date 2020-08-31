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


class DE_Network_Embedding_Regression(KerasRegression):
    def __init__(self, model, model_author, model_description, epochs=25, batch_size=1000, verbose=0):
        super(DE_Network_Embedding_Regression, self).__init__(model, model_author, model_description)
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def _fit(self, X, y):
        # checkpoint_filepath = 'sequence_only_cnn_{}.best.hdf5'.format(str(randint(1000000000, 9999999999)))
        # checkpoint_callback = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', save_best_only=True)
        # stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=3)
        # callbacks_list = [checkpoint_callback, stopping_callback]
        X.loc[:, 'gene_cat'] = pd.Categorical(X['gene'])
        X.loc[:, 'gene_cat'] = X.gene_cat.cat.codes
        X.loc[:, 'gene_2_cat'] = pd.Categorical(X['gene_2'])
        X.loc[:, 'gene_2_cat'] = X.gene_2_cat.cat.codes

        X1 = X['gene_cat']
        X2 = X['gene_2_cat']
        X3 = X.drop(['gene', 'gene_2', 'gene_cat', 'gene_2_cat'], axis=1)

        # Configure the output
        y_temp = pd.DataFrame(y.tolist(), index=y.index, columns=['logFC_col', 'edge_col'])
        y1 = y_temp['edge_col']
        y2 = y_temp['logFC_col']

        print("Length of ys", len(y1), len(y2))
        self.model.fit([X1, X2, X3], [y1, y2], epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        # self.model.load_weights(checkpoint_filepath)
        # os.remove(checkpoint_filepath)

    def _predict(self, X):
        X.loc[:, 'gene_cat'] = pd.Categorical(X['gene'])
        X.loc[:, 'gene_cat'] = X.gene_cat.cat.codes
        X.loc[:, 'gene_2_cat'] = pd.Categorical(X['gene_2'])
        X.loc[:, 'gene_2_cat'] = X.gene_2_cat.cat.codes

        X1 = X['gene_cat']
        X2 = X['gene_2_cat']
        X3 = X.drop(['gene', 'gene_2', 'gene_cat', 'gene_2_cat'], axis=1)

        y1, y2 = self.model.predict([X1, X2, X3])

        merged1 = list(itertools.chain(*y1))

        merged2 = list(itertools.chain(*y2))

        preds = list(zip(merged1, merged2))
        kk = [len(item) for item in preds]

        return preds


def DE_Embedding_Regression_with_network_reg(num_tokens, num_condition_cols, emb_dim=32, batch_size=10000, epochs=25,
                                             learning_rate=1e-3):
    # Gene 1 embedding layer
    input_row_1 = Input(shape=(1,))
    model1 = Embedding(input_dim=num_tokens, output_dim=emb_dim, input_length=1, batch_size=batch_size)(input_row_1)
    model1 = Flatten()(model1)
    model1 = Dense(4, input_dim=emb_dim, activation='relu')(model1)

    # Gene 2 embedding layer
    input_row_2 = Input(shape=(1,))
    model2 = Embedding(input_dim=num_tokens, output_dim=emb_dim, input_length=1, batch_size=batch_size)(input_row_2)
    model2 = Flatten()(model2)
    model2 = Dense(4, input_dim=emb_dim, activation='relu')(model2)

    # Combine gene layers and attach to output
    network_layer = Concatenate()([model1, model2])
    gene_network = Dense(1, activation='softmax', name='network_embedding')(network_layer)

    # Experiment condition input layer
    cond = Input(shape=(num_condition_cols,))
    model3 = Dense(4, input_dim=num_condition_cols, activation='relu')(cond)

    # Combine experiment layer with gene layer
    gene_cond = Concatenate()([model1, model3])
    de_prediction_layer = Dense(1, activation='linear', name='de_pred')(gene_cond)

    # Create model
    model = Model(inputs=[input_row_1, input_row_2, cond], outputs=[de_prediction_layer, gene_network])

    model.compile(loss={'network_embedding': 'binary_crossentropy',
                        'de_pred': 'mean_absolute_error'}, optimizer=Adam(lr=learning_rate),
                  metrics={'network_embedding': 'binary_crossentropy',
                           'de_pred': 'mse'})

    th_model = DE_Network_Embedding_Regression(model=model, model_author="Mohammed",
                                               model_description='DE Gene Embedding Model', batch_size=100, epochs=25)

    return th_model
