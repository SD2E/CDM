from harness.test_harness_models_abstract_classes import RegressionModel

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K


class CDM_Regression(RegressionModel):
    def __init__(self, model, model_author, model_description, epochs=25, batch_size=10000):
        super(CDM_Regression, self).__init__(model, model_author, model_description)
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_name = "CDM Regression"
        self.model_author = model_author
        self.model = model
        self.model_description = model_description

    def _fit(self, X, y):
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size)

    def _predict(self, X):
        preds = self.model.predict(X)
        K.clear_session()
        return preds


def CDM_regression_model(output_size, input_shape=None, batch_size=10000, epochs=25, learning_rate=1e-3):
    inputs = Input(shape=(input_shape,))

    x = Dense(input_shape, activation='relu')(inputs)
    x = Dense(input_shape * 6, activation='relu')(x)
    x = Dense(input_shape * 3, activation='relu')(x)
    x = Dense(output_size)(x)

    model = Model(inputs, outputs=x)
    model.compile(loss='mean_absolute_error', optimizer=Adam(lr=learning_rate))

    th_model = CDM_Regression(model=model, model_author="Jordan",
                              model_description="batch_size={0}, epochs={1}, learning_rate={2}".format(batch_size, epochs, learning_rate),
                              epochs=epochs, batch_size=batch_size)

    return th_model
