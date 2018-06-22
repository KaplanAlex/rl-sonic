
import math

from keras import backend as K
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, Add, Lambda, GaussianDropout
from keras.models import Sequential, Model
from keras.optimizers import Adam


class Networks(object):
    """
    Collection of networks used in various RL solution implementations.
    Featuring:
        DQN                 -  Deep Q-Network (Also can be used as Double DQN)
        Dueling DQN         -  DQN with Q(s,a) separated into V(s) and A(s,a)
        Noisy Dueling DQN   -  Dueling DQN with a Gaussian noise layer following
                               what was previously the output layer, allowing the
                               network to learn weights which reward or discourage
                               exploration.
                               

    """

    @staticmethod    
    def dqn(input_shape, action_size, learning_rate):
        """
        A convolutional neural network to approximate the q function.
        This network is comprised of 3 convolutional and pooling layers
        which culminate in a series of fully connected layers which
        ultimately output a value for each potential action.

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
    
    @staticmethod    
    def dueling_dqn(input_shape, action_size, learning_rate):
        """
        Neural network to approximate the q function similar to dqn.
        However, instead of directly approximating q values, this network
        breaks the value Q(s,a) - which respresents the value of being in
        state s and taking action a - into V(s) + A(s,a), which represent
        the value of being in state s and how much better action a is than 
        all other actions given state s respectively.  
        
        The final layer of the network sums the predicted V(s) and A(s,a) to
        yield the q value: Q(s,a) = V(s) + A(s,a). 

        Arguments:
            - input_shape: The dimensions of the input tensor 
              (height, width, channels).
            
            - action_size: The number of actions the agent can take. Dictates
              that number of nodes in the output layer.
            
            - learning_rate: The rate of change of the optimizer.
        """
        # Outline the network with the Keras functional API to specify the connection
        # of certain outputs to specific layers.
        input_layer = Input(shape=(input_shape))
        # Use the same convolutional layer outline (windows, strides, and pooling) as dqn.
        conv1 = Conv2D(32, kernel_size=(6, 6), strides=(1, 1),  activation='relu')(input_layer)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
        conv2 = Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu')(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
        conv3 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(pool2)
        conv_out = Flatten()(conv3)

        # Value - V(s)
        value_layer = Dense(256, activation='relu')(conv_out)
        value_layer = Dense(1, init='uniform')(value_layer)
        value_layer = Lambda(lambda s: K.expand_dims(s[:, 0], axis=-1), output_shape=(action_size,))(value_layer)

        # Advantage tower - A
        advantage_layer = Dense(256, activation='relu')(conv_out)
        advantage_layer = Dense(action_size)(advantage_layer)
        advantage_layer = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(action_size,))(advantage_layer)

        # merge the separate portions of the network to yield Q(s,a)
        action_value = Add()([value_layer, advantage_layer])
        
        model = Model(input=input_layer, output=action_value)
        
        adam = Adam(lr=learning_rate)
        model.compile(loss='mse',optimizer=adam)

        return model


    @staticmethod    
    def noisy_dueling_dqn(input_shape, action_size, learning_rate):
        """
        Dueling Double DQN (see "dueling_dqn") with the addition of a gaussian noise layer 
        to provide trainable exploration parameters. Noise is added to the generated
        Q values to add randomness to action selection. The weights on the noise addition
        are trained with the network, allowing exploration to be rewarded, and thus
        encouraged as the agent learns, and discouraged once the agent learns a reliable
        policy.

        Arguments:
            - input_shape: The dimensions of the input tensor 
              (height, width, channels).
            
            - action_size: The number of actions the agent can take. Dictates
              that number of nodes in the output layer.
            
            - learning_rate: The rate of change of the optimizer.
        """
        # Outline the network with the Keras functional API to specify the connection
        # of certain outputs to specific layers.
        input_layer = Input(shape=(input_shape))
        # Use the same convolutional layer outline (windows, strides, and pooling) as dqn.
        conv1 = Conv2D(32, kernel_size=(6, 6), strides=(1, 1),  activation='relu')(input_layer)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
        conv2 = Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu')(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
        conv3 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(pool2)
        conv_out = Flatten()(conv3)

        # Value - V(s)
        value_layer = Dense(256, activation='relu')(conv_out)
        value_layer = Dense(1, init='uniform')(value_layer)
        value_layer = Lambda(lambda s: K.expand_dims(s[:, 0], axis=-1), output_shape=(action_size,))(value_layer)

        # Advantage tower - A
        advantage_layer = Dense(256, activation='relu')(conv_out)
        advantage_layer = Dense(action_size)(advantage_layer)
        advantage_layer = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(action_size,))(advantage_layer)

        # merge the separate portions of the network to yield Q(s,a)
        action_value = Add()([value_layer, advantage_layer])
       
        # Deepmind paper suggests an implemntation of gaussian noise with standard
        # deviation = 1 / sqrt(# of inputs). GaussianDropout applies multiplicative noise
        # to weighted "action_value" 
        noise_weights = Dense(action_size, activation='linear')(action_value)
        noise = GaussianDropout(1 / math.sqrt(action_size))(noise_weights)

        # Sum action value with the noisy, weighted version of action value to allow
        # for the discouragement of noise (exploration) overtime.
        noisy_action_value = Add()([action_value, noise])

        model = Model(input=input_layer, output=noisy_action_value)
        
        adam = Adam(lr=learning_rate)
        model.compile(loss='mse',optimizer=adam)

        return model