import tensorflow as tf
import numpy as np
import pygeostat as gs
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from . lambda_distribution import beta_function
from sklearn.utils import shuffle
import pandas as pd

def keras_scatxval(model, data, features, labels):
    fig, ax = plt.subplots(nrows=1, ncols=len(labels), figsize=(5*len(labels),5))

    for i in range(len(labels)):
        prediction = model.predict(data[features])[:,i].flatten()
        true_value = data[labels].values[:,i].flatten()
        ax[i].set_title(labels[i])
        gs.scatxval(prediction, true_value, grid=True, ax=ax[i])
    plt.tight_layout()


def get_lambdas_ml(mean, variance, skewness, kurtosis, ml_model):

    '''
    A method to calculate lambda parameters for lambda distribution given the four first moments.
    The ml model (keras or tensor flow wrapper) is used to get skewness and kurtosis as features and
    estimate the expected lambda 3 and lambda 4. Lambda 1 and lambda 2 are then calculated mathematically.
    '''

    if isinstance(ml_model, Sequential):
        data = np.array([skewness, kurtosis])
        data = data.reshape(-1, 2)
        temp = ml_model.predict(data)
        lambda3 = temp[0][0]
        lambda4 = temp[0][1]
    else:
        temp = ml_model.get_prediction([skewness, kurtosis])
        lambda3 = temp[0]
        lambda4 = temp[1]

    v1_1 = lambda3 * ( lambda3 + 1 )
    v1_2 = lambda4 * ( lambda4 + 1 )
    v1 = (1/v1_1) - (1/v1_2)

    v2_1 = 1/((lambda3**2) * (2*lambda3 + 1))
    v2_2 = 1/((lambda4**2) * (2*lambda4 + 1))
    v2_3 = 2*beta_function(lambda3+1,lambda4+1)/(lambda3*lambda4)
    v2 = v2_1 + v2_2 - v2_3

    lambda2 = np.sqrt((v2-v1**2)/ variance)

    lambda1 = mean + (1/lambda2)* ( (1/(lambda3+1)) - (1/(lambda4+1)) )

    return lambda1, lambda2, lambda3, lambda4


class TensorFlowWrapper(object):

    def __init__(self, data_train, data_test, features, labels, seed):

        if not isinstance(data_train, pd.DataFrame):
            raise ValueError('data_train must be a Pandas DataFrame')

        if not isinstance(data_test, pd.DataFrame):
            raise ValueError('data_test must be a Pandas DataFrame')

        self.data_train = shuffle(data_train)
        self.data_train.reset_index(inplace=True, drop=True)
        self.data_test = data_test
        self.features = features
        self.labels = labels
        self.seed = seed

        self.n_features = len(features)
        self.n_output = len(labels)
        self.sess = tf.Session()


    def dispose(self):

        self.sess.close()


    def _convert2array(self):

        self.x_train = np.array(self.data_train[self.features], dtype=np.float32).reshape(len(self.data_train),self.n_features)
        self.y_train = np.array(self.data_train[self.labels], dtype=np.float32).reshape(len(self.data_train),self.n_output)

        self.x_test = np.array(self.data_test[self.features], dtype=np.float32).reshape(len(self.data_test),self.n_features)
        self.y_test = np.array(self.data_test[self.labels], dtype=np.float32).reshape(len(self.data_test),self.n_output)

    def get_prediction(self, data):
        raise NotImplementedError('This method was not implemented')


class MlpWrapper(TensorFlowWrapper):

    def __init__(self, data_train, data_test, features, labels, layer_n_nodes, seed=69069):

        super().__init__(data_train, data_test, features, labels, seed)

        if len(layer_n_nodes) == 0 or layer_n_nodes is None:
            self.layer_n_nodes = []
        else:
            self.layer_n_nodes = layer_n_nodes
        self.n_layers = len(layer_n_nodes)

        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=[self.n_layers+1], name='Keep_probability')
        self.weights = []
        self.biases = []
        self.layers = []
        self._convert2array()



    def create_computation_graph(self, weights_initializer = 'glorot_normal'):

        # Input layer
        with tf.name_scope(name='InputLayer'):
            # plcae holder for the input features
            self.x = tf.placeholder(dtype=tf.float32, shape=[None,self.n_features], name='InputFeatures')

        self.layers.append(tf.nn.dropout(self.x,self.keep_prob[0], seed=self.seed))

        # Desired label
        with tf.name_scope(name='Label'):
            #place holder for predicted output
            self.y = tf.placeholder(dtype=tf.float32, shape=[None,self.n_output], name = 'Label')

        # Get the proper weight initializer
        if weights_initializer.lower() == 'glorot_normal':
            initializer = tf.keras.initializers.glorot_normal(seed=self.seed) # Operation level seed

        elif weights_initializer.lower() == 'xavier':
            initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=self.seed)
        else:
            raise ValueError('Invalid weights_initializer')

        # Hidden layers
        previous_node = self.n_features
        node_num = previous_node
        for i, node_num in enumerate(self.layer_n_nodes):
            with tf.name_scope('Layer{}'.format(i+1)):

                self.weights.append(tf.Variable(initializer([previous_node,node_num]), dtype=tf.float32, name='Weights_L{}'.format(i+1)))
                self.biases.append(tf.Variable(tf.zeros([node_num]), dtype=tf.float32, name= 'Biases_L{}'.format(i+1)))
                layer = tf.matmul(self.layers[-1], self.weights[-1]) + self.biases[-1]
                layer = tf.nn.relu(layer)
                layer = tf.nn.dropout(layer,self.keep_prob[i+1], seed=self.seed)
                self.layers.append(layer)
                previous_node = node_num

        # Output layer
        with tf.name_scope('OutputLayer'):
            self.weights.append(tf.Variable(initializer([node_num,self.n_output]), dtype=tf.float32, name='Weights'))
            self.biases.append(tf.Variable(tf.zeros([self.n_output]), dtype=tf.float32, name= 'Biases'))
            self.y_p = tf.matmul(self.layers[-1],self.weights[-1]) + self.biases[-1]


    def compile(self, cost='MeanSquaredError', optimizer='AdamOptimizer',learning_rate=0.01, adaptive_learning_rate=False, decay_step=100, decay_rate=0.1):

        self.adaptive_learning_rate = adaptive_learning_rate
        if self.adaptive_learning_rate:
            self.initial_learning_rate = learning_rate
            self.learning_rate = tf.placeholder(tf.float32, shape=[])
            self.decay_rate =decay_rate
            self.decay_step = decay_step
        else:
            self.initial_learning_rate = learning_rate
            self.learning_rate = tf.placeholder(tf.float32, shape=[])
        # Cost
        if cost=='MeanSquaredError':
            with tf.name_scope('MeanSquaredError'):
                self.cost = tf.reduce_mean(tf.pow(tf.subtract(self.y_p, self.y),2))

        # Optimizer
        with tf.name_scope('Optimizer'):
            self.optimizer = tf.train.AdamOptimizer(name=optimizer,learning_rate=self.learning_rate).minimize(self.cost)


        # Performance measure
        if cost=='MeanSquaredError':
            with tf.name_scope(name='Performance'):
                self.mse = tf.reduce_mean((tf.pow(tf.subtract(self.y_p, self.y),2)) , name = 'MeanSquaredError')


    def fit(self, batch_size=200, nump_epoch=1000, report_interval = 10, keep_prob= 1, early_stop=False, patience=100, verbose=1):

        if not isinstance(keep_prob, list):
            try:
                keep_prob = [keep_prob]
            except:
                raise ValueError('Wrong keep probability')

        if len(keep_prob) == 1:
            keep_prob = [keep_prob[0] for i in range(self.n_layers+1)]
        elif len(keep_prob) != self.n_layers+1:
            raise ValueError('Keep probability must be one integer value or a list of integers for each layer including the input layer')
        keep_prob = np.array(keep_prob).reshape(self.n_layers+1,)
        self.keep_prob_test = np.ones(keep_prob.shape)

        batch_list = [ i for i in range(0,len(self.data_train),batch_size)] # Break training data into batches

        if verbose ==1:
            print('Total number of batches per epoch: {}'.format(len(batch_list)-1))
        if verbose ==2:
            for i in range(len(batch_list)-1):
                print('Random Batch {}: From {} to {}'.format(i+1, batch_list[i], batch_list[i+1]))

        # Implement a schedule for adaptive learning rate for the Gradient descent based optimization algorithm
        if self.adaptive_learning_rate:
            self.learning_rates = [self.initial_learning_rate * self.decay_rate**(i//nump_epoch) for i in range(nump_epoch)]
        else:
            self.learning_rates = [self.initial_learning_rate for i in range(nump_epoch)]


        try:
            self.dispose()
        except:
            pass
        self.sess = tf.Session()
        tf.random.set_random_seed(self.seed) # Graph level seed
        self.sess.run(tf.global_variables_initializer()) # Session initialization

        self.weights_nn=[]
        self.biases_nn = []
        self.cost_values = []

        for epoch in range(nump_epoch):
            for i in range(len(batch_list)-1):
                start_index = batch_list[i]
                end_index = batch_list[i+1]
                x_batch = self.x_train[start_index:end_index]
                y_batch = self.y_train[start_index:end_index]
                self.sess.run(self.optimizer, feed_dict={self.x: x_batch, self.y: y_batch, self.keep_prob : keep_prob, self.learning_rate : self.learning_rates[epoch]})


            cost_val = self.sess.run(self.cost, feed_dict={self.x: self.x_train, self.y: self.y_train , self.keep_prob :self.keep_prob_test})

            mse_test = self.sess.run(self.mse, feed_dict={self.x: self.x_test, self.y:self.y_test, self.keep_prob :self.keep_prob_test})

            self.cost_values.append(cost_val)

            if early_stop and epoch>patience+10:
              std_cost = np.std(self.cost_values[max(0,epoch-patience):epoch])
              cost_gap = np.abs(self.cost_values[0] - np.mean(self.cost_values[max(0,epoch-patience):epoch]))
              if (std_cost<0.01*cost_gap):
                  break

            # Keep track of changes in weights
            weight_temp = []
            for item in self.weights:
                weight_temp.append(item.eval(self.sess))
            self.weights_nn.append(weight_temp)

            biases_temp=[]
            for item in self.biases:
                biases_temp.append(item.eval(self.sess))
            self.biases_nn.append(biases_temp)

            if ((epoch+1)%10 ==0):
                print('Epoch: {}, Cost: {:.3f}, MSE_Test: {:.3f}'.format(epoch+1, cost_val, mse_test),end='')
                print('\r', end='')

    def get_prediction(self,data='test'):

        if isinstance(data, str):
            if data.lower() =='test':
                return self.sess.run(self.y_p, feed_dict = {self.x: self.x_test, self.keep_prob :self.keep_prob_test})
            else:
                return self.sess.run(self.y_p, feed_dict = {self.x: self.x_train, self.keep_prob :self.keep_prob_test})
        elif isinstance(data, np.ndarray):
            return self.sess.run(self.y_p, feed_dict = {self.x: data, self.keep_prob :self.keep_prob_test})
        else:
            try:
                data = np.array(data)
                data = data.reshape(-1, self.n_features)
                return self.sess.run(self.y_p, feed_dict = {self.x: data, self.keep_prob :self.keep_prob_test})[0]
            except:
                raise ValueError('Provided data should be either a string (test/train) or a numpy array')

    def scatxval(self, data='test', ax= None):

        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=self.n_output, figsize=(5*self.n_output,5))
            if self.n_output == 1:
                ax = [ax]

        predictions = self.get_prediction(data=data)

        for i in range(self.n_output):
            if data.lower() == 'test':
                true_value = self.y_test[:,i].flatten()
            else:
                true_value = self.y_train[:,i].flatten()
            prediction = predictions[:,i].flatten()
            ax[i].set_title(self.labels[i])
            gs.scatxval(prediction, true_value, grid=True, ax=ax[i])
        plt.tight_layout()

    def plot_cost_hist(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
        ax.plot(range(len(self.cost_values)), self.cost_values, lw=2, c='k')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cost')
        ax.grid(which='major', axis='y', linestyle='--')

    def save(self, model_url):
        import pickle
        pickle_url = '{model_url}.pickle'.format(model_url=model_url)
        with open(pickle_url, 'wb') as handle:
            container = MlpContainer(self.n_features, n_label = self.n_output, layers=self.layer_n_nodes, weights = self.weights_nn[-1], biases=self.biases_nn[-1])
            pickle.dump(container, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_container(model_url):
        import pickle
        pickle_url = '{model_url}.pickle'.format(model_url=model_url)
        with open(pickle_url, 'rb') as handle:
            return pickle.load(handle)

class MlpContainer(object):


    def __init__(self, n_features, n_label, layers, weights, biases):

        self.n_features = n_features
        self.n_label = n_label
        self.layers = layers
        self.weights = weights
        self.biases = biases

        if len(layers) + 1 != len(weights):
            raise ValueError('Inconsistent number of layers and weights')

        if len(layers) + 1 != len(biases):
            raise ValueError('Inconsistent number of layers and biases')


    def get_prediction(self, data):
        import numpy as np

        if not isinstance(data, np.ndarray):
            try:
                data = np.array(data)
                data = data.reshape(-1, self.n_features)
            except:
                raise ValueError('Provided data should be either a string (test/train) or a numpy array')


        output = data
        for weight, bias in zip(self.weights, self.biases):
            output = np.matmul(output, weight) + bias

        return output

