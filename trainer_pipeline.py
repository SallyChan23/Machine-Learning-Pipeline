# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# import tensorflow_transform as tft

# def _input_fn(file_pattern, tf_transform_output, batch_size=32):
#     transformed_feature_spec = tf_transform_output.transformed_feature_spec()
#     return tf.compat.v1.data.experimental.make_batched_features_dataset(
#     file_pattern=file_pattern,
#     batch_size=batch_size,
#     features=transformed_feature_spec,
#     label_key='charges_xf'
#     )

# def _build_keras_model(feature_keys):
#     inputs = {}
#     dense_inputs = []

#     for key in feature_keys:
#         if key == 'charges_xf':
#             continue

#         input_tensor = tf.keras.Input(shape=(1,), name=key, dtype=tf.float32, sparse=True)
#         inputs[key] = input_tensor

#         dense = tf.keras.layers.Lambda(lambda x: tf.expand_dims(tf.sparse.to_dense(x), -1))(input_tensor)
#         dense_inputs.append(dense)

#     x = tf.keras.layers.Concatenate()(dense_inputs)
#     x = tf.keras.layers.Dense(16, activation='relu')(x)
#     output = tf.keras.layers.Dense(1)(x)

#     model = tf.keras.Model(inputs=inputs, outputs=output)
#     model.compile(optimizer='adam',
#                   loss='mean_squared_error',
#                   metrics=['mae'])
#     return model

# def trainer_fn(fn_args, schema): 
#     tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    

#     train_dataset = _input_fn(fn_args.train_files, tf_transform_output)
#     eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output)

#     feature_keys = tf_transform_output.transformed_feature_spec().keys()
#     model = _build_keras_model(feature_keys)

#     model.fit(train_dataset,
#               validation_data=eval_dataset,
#               steps_per_epoch=fn_args.train_steps,
#               validation_steps=fn_args.eval_steps,
#               epochs=5)

#     model.save(fn_args.serving_model_dir, save_format='tf')

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_transform import TFTransformOutput
from tfx.components.trainer.executor import GenericExecutor


def _input_fn(file_pattern, tf_transform_output, batch_size=32):
    transformed_feature_spec = tf_transform_output.transformed_feature_spec()

    # Ganti jadi TFRecordDataset langsung + compression_type yang eksplisit
    raw_dataset = tf.data.TFRecordDataset(
        tf.io.gfile.glob(file_pattern),
        compression_type='GZIP'
    )

    def _parse_function(serialized_example):
        parsed_example = tf.io.parse_single_example(serialized_example, transformed_feature_spec)
        label = parsed_example.pop('charges_xf')
        return parsed_example, label

    def _dense_cast(x, y):
        result = {}
        for key, value in x.items():
            if isinstance(value, tf.SparseTensor):
                value = tf.sparse.to_dense(value)
            result[key] = tf.cast(value, tf.float32)

        if isinstance(y, tf.SparseTensor):
            y = tf.cast(tf.sparse.to_dense(y), tf.float32)
        else:
            y = tf.cast(y, tf.float32)

        return result, y

    return raw_dataset.map(_parse_function).map(_dense_cast).batch(batch_size)

# def _input_fn(file_pattern, tf_transform_output, batch_size=32):
#     transformed_feature_spec = tf_transform_output.transformed_feature_spec()

#     raw_dataset = tf.compat.v1.data.experimental.make_batched_features_dataset(
#         file_pattern=file_pattern,
#         batch_size=batch_size,
#         features=transformed_feature_spec,
#         label_key='charges_xf',
#         compression_type='GZIP'  
#     )

#     def _dense_cast(x, y):
#         result = {}
#         for key, value in x.items():
#             if isinstance(value, tf.SparseTensor):
#                 value = tf.sparse.to_dense(value)
#             result[key] = tf.cast(value, tf.float32)
        
#         if isinstance(y, tf.SparseTensor):
#             y = tf.cast(tf.sparse.to_dense(y), tf.float32)
#         else:
#             y = tf.cast(y, tf.float32)

#         return result, y

#     return raw_dataset.map(_dense_cast)

def _build_keras_model():
    age = tf.keras.Input(shape=(1,), name='age', dtype=tf.float32)
    bmi = tf.keras.Input(shape=(1,), name='bmi', dtype=tf.float32)
    children = tf.keras.Input(shape=(1,), name='children', dtype=tf.float32)

    sex = tf.keras.Input(shape=(1,), name='sex', dtype=tf.float32)
    smoker = tf.keras.Input(shape=(1,), name='smoker', dtype=tf.float32)
    region = tf.keras.Input(shape=(1,), name='region', dtype=tf.float32)

    x = tf.keras.layers.Concatenate()([
        age, bmi, children,
        sex, smoker, region
    ])

    x = tf.keras.layers.Dense(16, activation='relu')(x)
    output = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(
        inputs={
            'age': age,
            'bmi': bmi,
            'children': children,
            'sex': sex,
            'smoker': smoker,
            'region': region
        },
        outputs=output
    )

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mae'])
    return model

def run_fn(fn_args):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(fn_args.train_files, tf_transform_output)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output)

    model = _build_keras_model()

    model.fit(
        train_dataset,
        validation_data=eval_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_steps=fn_args.eval_steps,
        epochs=5
    )

    model.save(fn_args.serving_model_dir, save_format='tf')



# def trainer_fn(fn_args, schema):
#     print("🚀 Mulai trainer_fn")
#     print("🔍 fn_args.train_files:", fn_args.train_files)
#     print("🔍 fn_args.eval_files:", fn_args.eval_files)
#     print("🔍 fn_args.transform_output:", fn_args.transform_output)
#     tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

#     train_dataset = _input_fn(fn_args.train_files, tf_transform_output)
#     eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output)

#     model = _build_keras_model()

#     model.fit(
#         train_dataset,
#         validation_data=eval_dataset,
#         steps_per_epoch=fn_args.train_steps,
#         validation_steps=fn_args.eval_steps,
#         epochs=5
#     )

#     model.save(fn_args.serving_model_dir, save_format='tf')