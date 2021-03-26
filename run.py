import numpy as np
from source.utils import penchar_parser
from source.models import autoencoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

if __name__ == '__main__':
    label, data = penchar_parser()
    model, encoder_model, decoder_model = autoencoder(input_dim=2, output_dim=2, sequence_len=None, hidden_size=128)

    def train_generator():
        i = 0
        while True:
            if i >= len(data[:9312]):
                i = 0
            tmp_data = np.array(data[i]).astype('float')
            i += 1
            tmp_data = np.expand_dims(tmp_data, 0)
            yield tmp_data.copy(), tmp_data[:, ::-1, :].copy()

    def validation_generator():
        i = 0
        while True:
            if i >= len(data[9312:]):
                i = 0
            tmp_data = np.array(data[9312 + i]).astype('float')
            i += 1
            tmp_data = np.expand_dims(tmp_data, 0)
            yield tmp_data.copy(), tmp_data[:, ::-1, :].copy()
        
    es_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint_filepath = 'pen10000-{epoch:04d}.h5'
    ckp_callback = ModelCheckpoint(checkpoint_filepath)
    hist = model.fit_generator(train_generator(), steps_per_epoch=9312, epochs=100, validation_data=validation_generator(), validation_steps=2328, callbacks=[es_callback, ckp_callback])

