import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.get_weights())

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        flag = True
        size = 1
        while flag:
            flag = False
            for a, b in dataset.iterate_once(size):
                if nn.as_scalar(b) != self.get_prediction(a):
                    self.get_weights().update(a, nn.as_scalar(b))
                    flag = True

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.W1 = nn.Parameter(1, 50)
        self.W2 = nn.Parameter(50, 1)
        self.b1 = nn.Parameter(1, 50)
        self.b2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        prediction1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.W1), self.b1))
        final_prediction = nn.AddBias(nn.Linear(prediction1, self.W2), self.b2)
        return final_prediction

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        square_loss = nn.SquareLoss(self.run(x), y)
        return square_loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        alpha = -0.01
        loss = float("inf")
        while loss > 0.0100:
            for a, b in dataset.iterate_once(20):
                gradient_w1, gradient_w2, gradient_b1, gradient_b2 = nn.gradients(self.get_loss(a, b), [self.W1, self.W2, self.b1, self.b2])
                self.W1.update(gradient_w1, alpha)
                self.W2.update(gradient_w2, alpha)
                self.b1.update(gradient_b1, alpha)
                self.b2.update(gradient_b2, alpha)
                loss = nn.as_scalar(self.get_loss(a, b))


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.W1 = nn.Parameter(784, 400)
        self.W2 = nn.Parameter(400, 300)
        self.W3 = nn.Parameter(300, 10)
        self.b1 = nn.Parameter(1, 400)
        self.b2 = nn.Parameter(1, 300)
        self.b3 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        prediction1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.W1), self.b1))
        prediction2 = nn.ReLU(nn.AddBias(nn.Linear(prediction1, self.W2), self.b2))
        final_prediction = nn.AddBias(nn.Linear(prediction2, self.W3), self.b3)
        return final_prediction

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        softmaxloss = nn.SoftmaxLoss(self.run(x), y)
        return softmaxloss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        alpha = -0.99
        accuracy_threshold =  0.97
        while dataset.get_validation_accuracy() < accuracy_threshold:
            for a, b in dataset.iterate_once(1000):
                gradient_w1, gradient_w2, gradient_b1, gradient_b2, gradient_w3, gradient_b3, = nn.gradients(self.get_loss(a, b), [self.W1, self.W2, self.b1, self.b2, self.W3, self.b3])
                self.W1.update(gradient_w1, alpha)
                self.W2.update(gradient_w2, alpha)
                self.b1.update(gradient_b1, alpha)
                self.b2.update(gradient_b2, alpha)
                self.W3.update(gradient_w3, alpha)
                self.b3.update(gradient_b3, alpha)


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.W1 = nn.Parameter(47, 100)
        self.W2 = nn.Parameter(100, 100)
        self.W3 = nn.Parameter(100, 100)
        self.W4 = nn.Parameter(100, 100)
        self.W5 = nn.Parameter(100, 5)
        self.W6 = nn.Parameter(100, 100)
        self.b1 = nn.Parameter(1,100)
        self.b2 = nn.Parameter(1, 5)
        self.b3 = nn.Parameter(1, 100)
        self.b4 = nn.Parameter(1, 100)
        self.b5 = nn.Parameter(1, 5)
        self.b6 = nn.Parameter(1, 100)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        prediction_y1 = nn.ReLU(nn.AddBias(nn.Linear(xs[0], self.W1), self.b1))
        z_vector = nn.AddBias(nn.Linear(prediction_y1, self.W3), self.b3)

        for a in xs[1:]:
            prediction_y1 = nn.ReLU(nn.AddBias(nn.Linear(a, self.W1), self.b1))
            h_input1 = nn.AddBias(nn.Linear(prediction_y1, self.W3), self.b3)
            predicion_hidden1 = nn.ReLU(nn.AddBias(nn.Linear(z_vector, self.W4), self.b4))
            h_input2 = nn.AddBias(nn.Linear(predicion_hidden1, self.W6), self.b6)
            z_vector = nn.Add(h_input1, h_input2)
        final_y1 = nn.ReLU(nn.AddBias(nn.Linear(z_vector, self.W2), self.b1))
        z_vector = nn.AddBias(nn.Linear(final_y1, self.W5), self.b5)
        return z_vector
    

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        softmaxloss = nn.SoftmaxLoss(self.run(xs), y)
        return softmaxloss
    
    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        dimensions = 50
        hiddenDimensions =  250
        alpha = -0.05
        for i in range (0, dimensions):
            for a, b in dataset.iterate_once(hiddenDimensions):
                grad_wrt_w1, grad_wrt_w2, grad_wrt_b1, grad_wrt_b2, grad_wrt_w3, grad_wrt_b3, grad_wrt_w4, grad_wrt_b4, grad_wrt_w5, grad_wrt_b5, grad_wrt_w6, grad_wrt_b6 = \
                    nn.gradients(self.get_loss(a, b), [self.W1, self.W2, self.b1, self.b2, self.W3, self.b3, self.W4, self.b4, self.W5, self.b5, self.W6, self.b6])
                self.W1.update(grad_wrt_w1, alpha)
                self.W2.update(grad_wrt_w2, alpha)
                self.b1.update(grad_wrt_b1, alpha)
                self.b2.update(grad_wrt_b2, alpha)
                self.W3.update(grad_wrt_w3, alpha)
                self.b3.update(grad_wrt_b3, alpha)
                self.W4.update(grad_wrt_w4, alpha)
                self.b4.update(grad_wrt_b4, alpha)
                self.W5.update(grad_wrt_w5, alpha)
                self.b5.update(grad_wrt_b5, alpha)
                self.W6.update(grad_wrt_w6, alpha)
                self.b6.update(grad_wrt_b6, alpha)