import numpy as np
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras import backend as K

class Train():
    def __init__(self, model, batch_size=32, epochs=50, lrate=0.01):
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.lrate = lrate
        
    def compile_model(self, optimizer='sgd'):
        if optimizer == 'sgd':
            decay = self.lrate/self.epochs
            sgd = SGD(lr=self.lrate, momentum=0.9, decay=decay, nesterov=False)
            self.model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    def get_summary(self): 
        print self.model.summary()
    
    def train_model(self, xtrain, ytrain, xval, yval):
        self.model.fit(xtrain, ytrain, validation_data=(xval, yval), nb_epoch=self.epochs, batch_size=self.batch_size)

    def evaluate(self, xtest, ytest):
        scores = self.model.evaluate(xtest, ytest, verbose=0)
        return round(scores[1]*100, 3)

    def get_results(self, xtest, ytest):
        res = {'tp': 0, 'tn': 0, 'fn': 0, 'fp': 0}
        predictions = self.model.predict(xtest)

        for i in xrange(len(predictions)):
            pred = np.argmax(predictions[i])
            lab = np.argmax(ytest[i])

            if pred == 1 and lab == 1: res['tp'] += 1
            elif pred == 0 and lab == 0: res['tn'] += 1
            elif pred == 1 and lab == 0: res['fp'] += 1
            elif pred == 0 and lab == 1: res['fn'] += 1

        accuracy = (res['tp'] + res['tn']) / float(res['tp'] + res['tn'] + res['fp'] + res['fn'])
        sensitivity = (res['tp']) / float(res['tp'] + res['fn'])
        fpr = 1 - (res['tn']) / float(res['tn'] + res['fp'])
        return res, accuracy, sensitivity, fpr

        
        
