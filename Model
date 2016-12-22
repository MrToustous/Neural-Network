import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class neural_network_model():

    def __init__(self, layers_size = [], learning_rate = 0.01, n_epochs = 300, batch_size = 30):
        self.layers_size = layers_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        self.layers = []

    
    def predict(self, data):
        # Description de la partie "réseau de neurones" du graphe
        # Output layer i = Focntion_Activation(output_layer[i-1] * weights_layer[i] + bias_layer[i])
        for layer in self.layers[:-1]:
            data = tf.nn.relu(tf.add(tf.matmul(data, layer['weights']), layer['biases']))

        # La dernière couche ne contient pas de fonction d'activation
        return tf.add(tf.matmul(data, self.layers[-1]['weights']), self.layers[-1]['biases'])

    
    def fit(self, X_train, y_train, X_test = [], y_test = []):

        n_vars = len(X_train.columns)
        n_classes = len(y_train.columns)
        out_validation = False

        self.layers_size = [n_vars] + self.layers_size
        self.X = tf.placeholder(tf.float32, shape=[None, n_vars]) # 52 variables
        self.y = tf.placeholder(tf.float32, shape=[None, n_classes]) # 2 classes (buy or not)

        if ((len(X_test) != 0) & (len(y_test) != 0)):
            out_validation = True            

        # Initialisation des couches; initial weights can't be 0 for first backpropag. Random init.
        for i, size in enumerate(self.layers_size[1:]) :
            self.layers.append({'weights': tf.Variable(tf.random_normal([self.layers_size[i], size])), 
                                'biases': tf.Variable(tf.random_normal([size]))})

        self.layers.append({'weights': tf.Variable(tf.random_normal([self.layers_size[-1], n_classes])),
                            'biases': tf.Variable(tf.random_normal([n_classes]))})
        
        # Description de la partie "méta-paramètres" du graphe 
        prediction = self.predict(self.X) # Sortie de notre modèle
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, self.y)) # Fonction de coùt qui vient évaluer la performance du modèle
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost) # Optimise l'erreur grâce à AdamOptimizer (parametrable). Autres options: SGF, ADABoost,...
        # optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost) 
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(self.y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        predicted_proba = tf.contrib.learn.DNNClassifier.predict_proba

        l = np.arange(len(X_train))
        np.random.shuffle(l)

        epochs = []
        in_sample_error = []
        out_sample_error = []
        loss = []

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            for epoch in range(self.n_epochs):
                epoch_loss = 0
                
                for batch_index in [l[i:i+self.batch_size] for i in range(0, len(X_train), self.batch_size)]:
                    # epoch_X, epoch_y = tools.re_sampling(X_train, y_train, ratio = 1) # Coping with the unbalanced trainset
                    epoch_X, epoch_y = X_train.iloc[batch_index], y_train.iloc[batch_index]
                    _, c = sess.run([optimizer, cost], feed_dict = {self.X: epoch_X, self.y: epoch_y})
                    epoch_loss += c
                # print('Epoch', epoch + 1, 'completed out of', self.n_epochs, 'loss:', epoch_loss)
                # print('At Epoch', epoch + 1, 'Accuracy:', accuracy.eval(({self.X: X_train, self.y: y_train})))

                epochs.append(epoch)
                in_sample_error.append(accuracy.eval(({self.X: X_train, self.y: y_train})))
                if out_validation:
                    out_sample_error.append(accuracy.eval(({self.X: X_test, self.y: y_test})))
                loss.append(epoch_loss)

            plt.subplot(2, 1, 1)
            plt.xlabel('Epochs')
            plt.ylabel('In Sample Error')
            plt.plot(epochs, in_sample_error)
            if out_validation:
                plt.plot(epochs, out_sample_error)
            
            plt.subplot(2, 1, 2)
            plt.xlabel('Epochs')
            plt.ylabel('Loss Function')
            plt.plot(epochs, loss)
