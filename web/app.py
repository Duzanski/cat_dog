#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import base64
import numpy as np
import tensorflow as tf
import os
import sys

print(tf.__version__)

app = Flask(__name__)
api = Api(app)


class GetImage(Resource):
    def post(self):

        postedData = request.get_json()

        econded_image = postedData['image']
        econded_image = econded_image.encode('utf-8')

        with open('decoded.png', 'wb') as file_to_save:
            decoded_image = base64.decodebytes(econded_image)
            file_to_save.write(decoded_image)

        with open(os.path.join(sys.path[0], 'cnn_model.json'), 'r') as f:
            model_json = f.read()

            model = tf.keras.models.model_from_json(model_json)
            model.load_weights(os.path.join(sys.path[0], 'cnn_model.h5'))

        # with open('cnn_model.json', 'r') as f:
         #   model_json = f.read()

          #  model = tf.keras.models.model_from_json(model_json)
           # model.load_weights('cnn_model.h5')
        decoded_image = tf.keras.preprocessing.image.load_img(
            '/usr/src/app/decoded.png', target_size=(64, 64))
        decoded_image = tf.keras.preprocessing.image.img_to_array(
            decoded_image)
        decoded_image /= 255
        decoded_image = np.expand_dims(decoded_image, axis=0)

        previsao = model.predict(decoded_image)
        previsao = (previsao > 0.5)

        if previsao:
            retJson = {
                'status': 200,
                'tipo': 'Cachorro'
            }
        else:
            retJson = {
                'status': 200,
                'tipo': 'Gato'
            }

        return jsonify(retJson)


api.add_resource(GetImage, '/getimage')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
