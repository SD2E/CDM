from harness.test_harness_models_abstract_classes import ClassificationModel

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K


class CDM_Classification(ClassificationModel):
    def __init__(self, model, model_author, model_description, epochs=25, batch_size=10000):
        super(CDM_Classification, self).__init__(model, model_author, model_description)
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_name = "CDM Classification"
        self.model_author = model_author
        self.model = model
        self.model_description = model_description

    def _fit(self, X, y):
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size)

    def _predict(self, X):
        preds = self.model.predict(X)
        # K.clear_session()
        return preds

    def _predict_proba(self, X):
        probs = self.model.predict(X)
        K.clear_session()
        return probs.max(axis=1)

def CDM_classification_model(input_size, output_size, batch_size=10000,epochs=25,learning_rate=1e-3):

    inputs = Input(shape=(input_size,))

    x = Dense(input_size, activation='relu')(inputs)
    x = Dense(input_size*6, activation='relu')(x)
    x = Dense(input_size*3, activation='relu')(x)
    x = Dense(output_size, activation='softmax')(x)

    model = Model(inputs, outputs=x)
    model.compile(loss='mean_absolute_error', optimizer=Adam(lr=learning_rate))

    th_model = CDM_Classification(model=model, model_author="Jordan", model_description="batch_size={0}, epochs={1}, learning_rate={2}".format(batch_size, epochs, learning_rate),epochs=epochs, batch_size=batch_size)

    return th_model