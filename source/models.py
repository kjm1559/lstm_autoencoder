import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow as tf

def repeat_vector(args):
    '''
    Adaptive sequence repeat vector
    Arg:
        args: argments
    Returns:
        layer (keras.layers): repeat_vector layer
    '''
    layer_to_repeat = args[0]
    sequence_layer = args[1]
    return keras.layers.RepeatVector(K.shape(sequence_layer)[1])(layer_to_repeat)
    
def encoder(input_dim, sequence_len, hidden_size):
    '''
    lstm encoder
    Arg:
        input_dim (int): input dimension
        output_dim (int): output dimension
        sequence_len (int): time sequence lenght
        hidden_size (int): hidden size
    Returns:
        model (keras.Model): keras model
    '''
    inputs = keras.Input(shape=(sequence_len, input_dim))
    x = keras.layers.TimeDistributed(keras.layers.BatchNormalization(axis=-1))(inputs)
    x = keras.layers.TimeDistributed(keras.layers.Dropout(0.1))(x)
    x = keras.layers.LSTM(hidden_size, return_sequences=True)(x)
    latent = keras.layers.LSTM(hidden_size)(x)
    return keras.Model(inputs, latent)

def decoder(output_dim, sequence_len, hidden_size):
    '''
    lstm decoder
    Arg:
        input_dim (int): input dimension
        output_dim (int): output dimension
        sequence_len (int): time sequence lenght
        hidden_size (int): hidden size
    Returns:
        model (keras.Model): keras model
    '''
    inputs = keras.Input(shape=(sequence_len, hidden_size,))
    
    x = keras.layers.LSTM(hidden_size, return_sequences=True)(inputs)
    x = keras.layers.LSTM(hidden_size, return_sequences=True)(x)
    
    # dense layers
    x = keras.layers.TimeDistributed(keras.layers.Dense(512, activation='relu'))(x)
    x = keras.layers.TimeDistributed(keras.layers.Dropout(0.2))(x)
    
    x = keras.layers.TimeDistributed(keras.layers.Dense(128, activation='relu'))(x)
    x = keras.layers.TimeDistributed(keras.layers.Dropout(0.2))(x)
        
    recon = keras.layers.TimeDistributed(keras.layers.Dense(128, activation='relu'))(x)
    
    recon = keras.layers.TimeDistributed(keras.layers.Dense(output_dim, activation='linear'))(x)
    return keras.Model(inputs, recon)

def mse_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))

def autoencoder(input_dim=2, output_dim=2, sequence_len = None, hidden_size=128):
    '''
    autoencoder 
    Arg:
        input_dim (int): input dimension
        output_dim (int): output dimension
        sequence_len (int): time sequence lenght
        hidden_size (int): hidden size
    Returns:
        model (keras.Model): keras model
    '''
    # input data
    inputs = keras.Input(shape=(None, input_dim))

    encoder_model = encoder(input_dim, sequence_len, hidden_size)
    decoder_model = decoder(output_dim, sequence_len, hidden_size)
    
    latent = encoder_model(inputs)
    decoder_input = keras.layers.Lambda(repeat_vector, output_shape=(sequence_len, hidden_size)) ([latent, inputs])
    outputs = decoder_model(decoder_input)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    recon_loss = K.mean(keras.losses.mse(inputs[:, ::-1, :], outputs), axis=-1)
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-3), loss=mse_loss)#, metric='mse')
    return model, encoder_model, decoder_model
    