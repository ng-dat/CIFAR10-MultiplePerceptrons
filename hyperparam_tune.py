from matplotlib import pyplot as plt
import numpy as np
import pickle


def get_data(inputs_file_path):
    """
    Takes in an inputs file path and labels file path, loads the data into a dict,
    normalizes the inputs, and returns (NumPy array of inputs, NumPy
    array of labels).
    :param inputs_file_path: file path for ONE input batch, something like
    'cifar-10-batches-py/data_batch_1'
    :return: NumPy array of inputs as float32 and labels as int8
    """
    # TODO: Load inputs and labels
    file = open(inputs_file_path, 'rb')
    raw_data = pickle.load(file, encoding='bytes')
    inputs = raw_data[b'data'].astype('float32')
    labels = np.array(raw_data[b'labels'], dtype='int8')
    # TODO: Normalize inputs
    inputs /= 255.0
    return inputs, labels


class Model:
    """
    This model class will contain the architecture for
    your single layer Neural Network for classifying CIFAR10 with
    batched learning. Please implement the TODOs for the entire
    model but do not change the method and constructor arguments.
    Make sure that your Model class works with multiple batch
    sizes. Additionally, please exclusively use NumPy and
    Python built-in functions for your implementation.
    """

    def __init__(self):
        # TODO: Initialize all hyperparametrs
        self.input_size = 3 * 32 * 32  # 3072 # Size of image vectors
        self.num_classes = 10  # Number of classes/possible labels
        self.batch_size = 100  # recommended default 100
        self.learning_rate = 0.005  # recommended default

        # TODO: Initialize weights and biases
        self.W = np.zeros((self.num_classes, self.input_size))
        self.b = np.zeros(self.num_classes)

    def forward(self, inputs):
        """
        Does the forward pass on an batch of input images.
        :param inputs: normalized (0.0 to 1.0) batch of images,
                       (batch_size x 3072) (2D), where batch can be any number.
        :return: probabilities, probabilities for each class per image # (batch_size x 10)
        """
        # TODO: Write the forward pass logic for your model
        Z = inputs @ self.W.T
        Z[:] += self.b

        # TODO: Calculate, then return, the probability for each class per image using the Softmax equation
        def softmax(z):
            e_z = np.exp(z - np.max(z))
            return e_z / e_z.sum()

        probabilities = np.apply_along_axis(softmax, 1, Z)
        return probabilities

    def loss(self, probabilities, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Loss should be generally decreasing with every training loop (step).
        :param probabilities: matrix that contains the probabilities
        of each class for each image
        :param labels: the true batch labels
        :return: average loss per batch element (float)
        """
        # TODO: calculate average cross entropy loss for a batch
        N = labels.shape[0]
        # Get one-hot y. (?) Question: labels is in one-hot or not? -> Assume that they are not in one-hot yet
        one_hot_y = np.zeros((N, self.num_classes))
        for n in range(N):
            one_hot_y[n, labels[n]] = 1.0
        # Get entropy loss
        loss = -np.mean(np.sum(one_hot_y * np.log(probabilities), axis=1))
        # Return
        return loss

    def compute_gradients(self, inputs, probabilities, labels):
        """
        Returns the gradients for model's weights and biases
        after one forward pass and loss calculation. You should take the
        average of the gradients across all images in the batch.
        :param inputs: batch inputs (a batch of images)
        :param probabilities: matrix that contains the probabilities of each
        class for each image
        :param labels: true labels
        :return: gradient for weights,and gradient for biases
        """
        # TODO: calculate the gradients for the weights and the gradients for the bias with respect to average loss

        N = labels.shape[0]
        # Get one-hot y. (?) Question: labels is in one-hot or not? -> Assume that they are not in one-hot yet
        one_hot_y = np.zeros((N, self.num_classes))
        for n in range(N):
            one_hot_y[n, labels[n]] = 1.0
        # Get gradients
        grad_W = (probabilities - one_hot_y).T @ inputs / N
        grad_b = np.mean(probabilities - one_hot_y, axis=0)
        # Return
        return grad_W, grad_b

    def accuracy(self, probabilities, labels):
        """
        Calculates the model's accuracy by comparing the number
        of correct predictions with the correct answers.
        :param probabilities: result of running model.forward() on test inputs
        :param labels: test set labels
        :return: Float (0,1) that contains batch accuracy
        """
        # TODO: calculate the batch accuracy
        # (?) Question: description of function is so confusing.
        #           - What means "comparing the number of correct predictions with the correct answers"? Why "correct prediction" is different from "correct answers"?
        #           - The "accuracy" in Algorithm1 pseudocode takes the parameters that is from only 1 sample at time. But this function takes multiple sample
        #           - Also that function takes the predicted label as paramater while this function takes predicted probabilities?
        predicted_labels = np.argmax(probabilities, axis=1)
        accuracy = np.mean(predicted_labels == labels)
        return accuracy

    def gradient_descent(self, gradW, gradB):
        '''
        Given the gradients for weights and biases, does gradient
        descent on the Model's parameters.
        :param gradW: gradient for weights
        :param gradB: gradient for biases
        :return: None
        '''
        # TODO: change the weights and biases of the model to descent the gradient
        self.W -= gradW * self.learning_rate
        self.b -= gradB * self.learning_rate


def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels.
    :param model: the initialized model to use for the forward
    pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training)
    :param train_inputs: train labels (all labels to use for training)
    :return: None
    '''
    # TODO: Iterate over the training inputs and labels, in model.batch_size increments
    N = train_labels.shape[0]
    batch_num = (N + 1) // model.batch_size
    losses = []
    for batch_idx in range(batch_num):
        batch_inputs = train_inputs[batch_idx * model.batch_size: (batch_idx + 1) * model.batch_size, :]
        batch_labels = train_labels[batch_idx * model.batch_size: (batch_idx + 1) * model.batch_size]
        # TODO: For every batch, compute then descend the gradients for the model's weights
        predicted_prob = model.forward(batch_inputs)
        grad_W, grad_b = model.compute_gradients(batch_inputs, predicted_prob, batch_labels)
        model.gradient_descent(grad_W, grad_b)


def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels.
    :param test_inputs: CIFAR10 test data (all images to be tested)
    :param test_labels: CIFAR10 test labels (all corresponding labels)
    :return: accuracy - Float (0,1)
    """
    # TODO: Iterate over the testing inputs and labels
    # (?) Question: can we use vector instead of running for loop? -> Assume yes
    predicted_prob = model.forward(test_inputs)
    # TODO: Return accuracy across testing set
    return model.accuracy(predicted_prob, test_labels)


def tune():
    train_inputs, train_labels = get_data('cifar-10-batches-py/data_batch_1')
    for i in range(2, 5):
        batch_i_images, batch_i_labels = get_data('cifar-10-batches-py/data_batch_' + str(i))
        train_inputs = np.append(train_inputs, batch_i_images, axis=0)
        train_labels = np.append(train_labels, batch_i_labels, axis=0)
    val_inputs, val_labels = get_data('cifar-10-batches-py/data_batch_5')

    batch_sizes = [10,50,80,100,200,500,1000,1500,2000]
    learning_rates = [0.1,0.01,0.005,0.004,0.002,0.0015,0.001,0.0005]

    for b in batch_sizes:
        for l in learning_rates:
            model = Model()
            model.batch_size = b
            model.learning_rate = l
            train(model, train_inputs, train_labels)
            val_acc = test(model, val_inputs, val_labels)
            print('bs',b, 'lr',l, 'acc', val_acc)

def get_train_val(val_index):
    first_train_index = 1 if val_index != 1 else 2
    train_inputs, train_labels = get_data('cifar-10-batches-py/data_batch_'+str(first_train_index))
    for i in range(1,6):
        if i == first_train_index or i == val_index:
            continue
        batch_i_images, batch_i_labels = get_data('cifar-10-batches-py/data_batch_' + str(i))
        train_inputs = np.append(train_inputs, batch_i_images, axis=0)
        train_labels = np.append(train_labels, batch_i_labels, axis=0)
    val_inputs, val_labels = get_data('cifar-10-batches-py/data_batch_'+str(val_index))
    return train_inputs, train_labels, val_inputs, val_labels

def cross_val_tune():
    batch_sizes = [2]#[10,50,80,100]
    learning_rates = [0.01,0.005,0.004,0.002,0.0015,0.001]

    for b in batch_sizes:
        for l in learning_rates:
            val_acc = 0
            for val_index in range(1,6):
                train_inputs, train_labels, val_inputs, val_labels = get_train_val(val_index)
                model = Model()
                model.batch_size = b
                model.learning_rate = l
                train(model, train_inputs, train_labels)
                val_acc += test(model, val_inputs, val_labels)
            val_acc /= 5
            print('bs',b, 'lr',l, 'acc', val_acc)


if __name__ == '__main__':
    #tune()
    cross_val_tune()
