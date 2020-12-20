import tensorflow as tf
import tensorboard

def kerasModel(x_train, y_train):
	n_inputs = x_train.shape[1]
	n_labels = y_train.shape[1] if len(y_train.shape)>1 else 1


	NN_SHAPE = [n_inputs, 10, n_labels]
	layers = []
	for i in range(len(NN_SHAPE)-1):
		if i == 0:
			layers.append(tf.keras.layers.Dense(NN_SHAPE[i+1], input_shape = (NN_SHAPE[i],), activation="sigmoid"))
		else:
			layers.append(tf.keras.layers.Dense(NN_SHAPE[i+1], activation = "sigmoid"))

	n_layers = len(layers)
	model = tf.keras.Sequential(layers, name="NNModel1")


	## optimizer
	sgd = tf.keras.optimizers.SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
	ada = tf.keras.optimizers.Adagrad(learning_rate=0.01)


	# compile(optimizer, loss=None, metrics=None, 
	# loss_weights=None, sample_weight_mode=None, 
	# weighted_metrics=None, target_tensors=None)
	model.compile(loss=tf.keras.losses.mean_squared_error,
				optimizer=ada,
				metrics=["accuracy"],
				)

	fit = model.fit(x_train, y_train,
				batch_size       = 5,
				epochs           = 10,
				validation_split = 0.1,
				verbose          = 0,
				shuffle          = True)
	
	return model

# evaluation = model.evaluate(x=x_test, y=y_test, batch_size=None, verbose=0)

# predict(x, batch_size=None, verbose=0, steps=None, callbacks=None, 
# max_queue_size=10, workers=1, use_multiprocessing=False)
# pred_layer = model.predict(x_test, batch_size=None, verbose=0)

