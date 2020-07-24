import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Input, concatenate, Masking, LSTM, TimeDistributed, Bidirectional
from tensorflow.keras import regularizers 
from tensorflow.keras import initializers

def double_LSTM_hyper1():
    jet_input = Input(shape=(trainX_jets.shape[1], trainX_jets.shape[2]))
    Mask = Masking(-2)(jet_input)
    LSTM11 = LSTM(224, return_sequences=True)(Mask)
    LSTM12 = LSTM(224, return_sequences=True)(LSTM11)
    
    flat_jets = Flatten()(LSTM12)
    
    other_input = Input(shape=(trainX_other.shape[1]))
    Dense21 = Dense(224, activation='relu')(other_input)
    flat_other = Flatten()(Dense21)
    
    concat = concatenate([flat_other, flat_jets])
    dense1 = Dense(224, activation='relu')(concat)
    dense2 = Dense(224, activation='relu')(dense1)
    output = Dense(len(Y_names), activation='linear')(dense2)
    
    model = keras.models.Model(inputs=[jet_input, other_input], outputs=output)
    optimizer = keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(loss='mse', optimizer= optimizer, metrics=['mse'])
    
    return model 


def jet_weights():
    jet_input = Input(shape=(trainX_jets.shape[1], trainX_jets.shape[2]))
    Mask = Masking(-2)(jet_input)
    other_input = Input(shape=(trainX_other.shape[1]))
    flat_jets =  Flatten()(jet_input)
    concat0 = concatenate([other_input, flat_jets])
    PreDense1 = Dense(128, activation='tanh')(concat0)
    PreDense2 = Dense(64, activation='tanh')(PreDense1)
    PreDense3 = Dense(trainX_jets.shape[1], activation='softmax')(PreDense2)
    Norm = Lambda(lambda x: tf.math.multiply(x, 4))(PreDense3) # 4 relevant jets
    Shape_Dot = Reshape((-1,1))(Norm)
    Dot_jets = Multiply()([Shape_Dot, Mask])
    
    TDDense11 = TimeDistributed(Dense(64, activation='relu'))(Dot_jets)
    TDDense12 = TimeDistributed(Dense(64, activation='relu'))(TDDense11)
    Sum = Flatten()(TDDense12)
    # Sum = Lambda(lambda x: tf.reduce_sum(x,1))(TDDense12)
    Dense13 = Dense(64, activation='relu')(Sum)
    flat_right = Flatten()(Dense13)
    
    
    Dense21 = Dense(64, activation='relu')(other_input)
    Dense22 = Dense(64, activation='relu')(Dense21)
    flat_other = Flatten()(Dense22)
    
    concat = concatenate([flat_other, flat_right])
    dense1 = Dense(256, activation='relu')(concat)
    dense2 = Dense(128, activation='relu')(dense1)
    output = Dense(len(Y_names), activation='linear')(dense2)
    
    model = keras.models.Model(inputs=[jet_input, other_input], outputs=output)
    optimizer = keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(loss='mse', optimizer= optimizer, metrics=['mse'])
    
    return model 
