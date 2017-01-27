from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras import backend as K

class Train():
    def __init__(self, model, batch_size=32, epochs=50, lrate=0.01):
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.lrate = lrate
        
    def compile_model(self):
        decay = self.lrate/self.epochs
        sgd = SGD(lr=self.lrate, momentum=0.9, decay=decay, nesterov=False)
        self.model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    def get_summary(self): 
        print self.model.summary()
    
    def train_model(self, xtrain, xval, ytrain, yval):
        self.model.fit(xtrain, ytrain, validation_data=(xval, yval), nb_epochs=self.epochs, batch_size=self.batch_size)
    
    def evaluate(self, xtest, ytest):
        scores = self.model.evaluate(xtest, ytest, verbose=0)
        return round(scores[1]*100, 3)
