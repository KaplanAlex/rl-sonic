

from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.optimizers import Adam


class Networks(object):
    """
    Collection of networks used in various RL solution  implementations.
    Featuring:
        DQN - Deep Q-Network
    """

    @staticmethod    
    def dqn(input_shape, action_size, learning_rate):
        """
        A convolutional neural network to approximate the q function relating
        (state, action) pairs to 

        Arguments:
            - input_shape: The dimensions of the input tensor 
              (height, width, channels).
            
            - action_size: The number of actions the agent can take. Dictates
              that number of nodes in the output layer.
            
            - learning_rate: The rate of change of the optimizer.
        """
        model = Sequential()
        # Start with a convolutional layer with defined input shape...
        # 32 output Filters each with a window of 6x6 with stride 1. 
        # (4D output - stack of 2d convolution results for each filter for each input sample in the batch)
        model.add(Conv2D(32, kernel_size=(6, 6), strides=(1, 1), activation='relu', input_shape=input_shape))
        # Pool to extract features, reduce dimensionality, and generalize.
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # Second convolutional layer with 64 filters followed by pooling
        model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # Final convolutional layer.
        model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        # Flatten the 3D output of the 64 filters with features from 2D images to a 1D vector.
        model.add(Flatten())
        # Two dense layers to progressively reach a final output size of action_size.
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(action_size, activation='linear'))
        # Adam optimzer on mse loss to apply updates to q values based on bellman's equation
        # Q(s, a) = r + ymax(a)(Q(s'a'))
        adam = Adam(lr=learning_rate)
        model.compile(loss='mse', optimizer=adam)

        return model
