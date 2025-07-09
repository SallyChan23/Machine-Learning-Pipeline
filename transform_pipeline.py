# import tensorflow_transform as tft

# def preprocessing_fn(inputs):
#     outputs = {}
#     outputs['age'] = tft.scale_to_z_score(inputs['age'])
#     outputs['bmi'] = tft.scale_to_z_score(inputs['bmi'])
#     outputs['children'] = inputs['children']
#     outputs['sex'] = tft.compute_and_apply_vocabulary(inputs['sex'])
#     outputs['smoker'] = tft.compute_and_apply_vocabulary(inputs['smoker'])
#     outputs['region'] = tft.compute_and_apply_vocabulary(inputs['region'])

#     outputs['charges_xf'] = tft.scale_to_z_score(inputs['charges'])  
#     return outputs

import tensorflow_transform as tft
import tensorflow as tf

def preprocessing_fn(inputs):
    outputs = {}
    outputs['age'] = tft.scale_to_z_score(inputs['age'])
    outputs['bmi'] = tft.scale_to_z_score(inputs['bmi'])
    outputs['children'] = tf.cast(inputs['children'], tf.float32)  
    outputs['sex'] = tf.cast(tft.compute_and_apply_vocabulary(inputs['sex']), tf.float32)
    outputs['smoker'] = tf.cast(tft.compute_and_apply_vocabulary(inputs['smoker']), tf.float32)
    outputs['region'] = tf.cast(tft.compute_and_apply_vocabulary(inputs['region']), tf.float32)
    outputs['charges_xf'] = tft.scale_to_z_score(inputs['charges'])
    return outputs