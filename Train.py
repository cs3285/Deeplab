

def parse_function_train(filename, filefolder, labelfolder):
	image_string = tf.read_file(filefolder + "\\" + filename)
	image_decoded = tf.image.decode_jpeg(image_string, channels = 1)
	image = tf.image.convert_image_dtype(image_decoded, tf.float32)
	label_string = tf.read_file(labelfolder + "\\" + filename)
	label_decoded = tf.image.decode_jpeg(label_string, channels = 1)
	label = tf.image.convert_image_dtype(label_decoded, tf.int32)
	return image, label

	

def input_fn(is_training, filefolder, params, labelfolder = None):
	filenames = os.listdir(filefolder)
	parse_fn_train = lambda f: parse_function_train(f, filefolder, labelfolder)
	parse_fn_test = lambda f: parse_function_test(f, filefolder)
	if is_training:
		dataset = (tf.data.Dataset.from_tensor_slices(filenames)
			.shuffle(len(filenames))
			.map(parse_fn_train, num_parallel_calls = params["num_parallel_calls"])
			#.map(preprocess_fn, num_parallel_calls = paras["num_parallel_calls"])
			.batch(params["batch_size"])
			.prefetch(1)
			)
	else:
		dataset = (tf.data.Dataset.from_tensor_slices(filenames)
			.map(parse_fn_test)
			.batch(params.batch_size)
			.prefetch(1)
			)
	iterator = dataset.make_initializable_iterator()
	images, labels = iterator.get_next()
	iterator_init_op = iterator.initializer

	inputs = {"images": images, "labels": labels, "iterator_init_op": iterator_init_op}
	return inputs


def train():

	train_inputs = input_fn

