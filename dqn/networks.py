import tensorflow as tf

from tf.keras.models import Sequential, load_model, Model


class Networks(object):

    @staticmethod    
    def dqn(input_shape, action_size, learning_rate):
        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4,4), activation='relu', input_shape=(input_shape)))
        model.add(Convolution2D(64, 4, 4, subsample=(2,2), activation='relu'))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(output_dim=512, activation='relu'))
        model.add(Dense(output_dim=action_size, activation='linear'))

        adam = Adam(lr=learning_rate)
        model.compile(loss='mse',optimizer=adam)

        return model
    
    @staticmethod    
    def dueling_dqn(input_shape, action_size, learning_rate):

        state_input = Input(shape=(input_shape))
        x = Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu')(state_input)
        x = Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu')(x)
        x = Convolution2D(64, 3, 3, activation='relu')(x)
        x = Flatten()(x)

        # state value tower - V
        state_value = Dense(256, activation='relu')(x)
        state_value = Dense(1, init='uniform')(state_value)
        state_value = Lambda(lambda s: K.expand_dims(s[:, 0], dim=-1), output_shape=(action_size,))(state_value)

        # action advantage tower - A
        action_advantage = Dense(256, activation='relu')(x)
        action_advantage = Dense(action_size)(action_advantage)
        action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(action_size,))(action_advantage)

        # merge to state-action value function Q
        state_action_value = merge([state_value, action_advantage], mode='sum')

        model = Model(input=state_input, output=state_action_value)
        #model.compile(rmsprop(lr=learning_rate), "mse")
        adam = Adam(lr=learning_rate)
        model.compile(loss='mse',optimizer=adam)

        return model

    @staticmethod    
    def drqn(input_shape, action_size, learning_rate):

        model = Sequential()
        model.add(TimeDistributed(Convolution2D(32, 8, 8, subsample=(4,4), activation='relu'), input_shape=(input_shape)))
        model.add(TimeDistributed(Convolution2D(64, 4, 4, subsample=(2,2), activation='relu')))
        model.add(TimeDistributed(Convolution2D(64, 3, 3, activation='relu')))
        model.add(TimeDistributed(Flatten()))

        # Use all traces for training
        #model.add(LSTM(512, return_sequences=True,  activation='tanh'))
        #model.add(TimeDistributed(Dense(output_dim=action_size, activation='linear')))

        # Use last trace for training
        model.add(LSTM(512,  activation='tanh'))
        model.add(Dense(output_dim=action_size, activation='linear'))

        adam = Adam(lr=learning_rate)
        model.compile(loss='mse',optimizer=adam)

        return model