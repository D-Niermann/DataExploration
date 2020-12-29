import tensorflow as tf
import numpy as np
import tensorboard

def kerasModel(x_train : np.array, y_train : np.array, NNShape : list) -> tf.python.keras.engine.sequential.Sequential:
	n_inputs = x_train.shape[1]
	n_labels = y_train.shape[1] if len(y_train.shape)>1 else 1


	layers = []
	for i in range(len(NNShape)-1):
		if i == 0:
			layers.append(tf.keras.layers.Dense(NNShape[i+1], 	input_shape = (NNShape[i],),
																activation = "sigmoid"))
		else:
			layers.append(tf.keras.layers.Dense(NNShape[i+1], activation = "sigmoid"))

	n_layers = len(layers)
	model = tf.keras.Sequential(layers, name="NNModel1")


	## optimizer
	sgd = tf.keras.optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
	ada = tf.keras.optimizers.Adagrad(learning_rate=0.2)


	# compile(optimizer, loss=None, metrics=None, 
	# loss_weights=None, sample_weight_mode=None, 
	# weighted_metrics=None, target_tensors=None)
	model.compile(loss=tf.keras.losses.mean_squared_error,
				optimizer=ada,
				metrics=["accuracy"],
				)

	fit = model.fit(x_train, y_train,
				batch_size       = 5,
				epochs           = 50,
				validation_split = 0.1,
				verbose          = 0,
				shuffle          = False)
	
	return model

# evaluation = model.evaluate(x=x_test, y=y_test, batch_size=None, verbose=0)

# predict(x, batch_size=None, verbose=0, steps=None, callbacks=None, 
# max_queue_size=10, workers=1, use_multiprocessing=False)
# pred_layer = model.predict(x_test, batch_size=None, verbose=0)

