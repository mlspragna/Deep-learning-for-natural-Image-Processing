from keras import Input, layers
from keras import optimizers
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Embedding, Dense, Activation, Flatten, Reshape, Dropout
from keras.layers import add,concatenate
from keras.models import Model

def image_caption_consistency_model(max_sen_length, vocab_size, embedding_matrix):
	## Image feature extraction model
	inputs1 = Input(shape=(2048,))
	hidden_layer = Dropout(0.5)(inputs1)
	image_features = Dense(256, activation='relu')(hidden_layer)

	## Captions feature extraction model
	inputs2 = Input(shape=(max_sen_length,))
	hidden_layer1 = Embedding(vocab_size, 200, mask_zero=True)(inputs2)
	hidden_layer2 = Dropout(0.5)(hidden_layer1)
	caption_features = LSTM(256)(hidden_layer2)

	## FFNN model
	decoder1 = concatenate([image_features, caption_features])
	decoder2 = Dense(256, activation='relu')(decoder1)
	outputs = Dense(1, activation='sigmoid')(decoder2)

	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.layers[2].set_weights([embedding_matrix])
	model.layers[2].trainable = False

	print(model.summary())

	return model
