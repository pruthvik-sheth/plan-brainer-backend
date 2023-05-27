import cv2
import base64
import numpy as np
from io import BytesIO
from tensorflow import convert_to_tensor, cast, float32
from flask_cors import CORS
from keras.models import load_model
from flask import Flask, render_template, Response, abort, request, jsonify

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

model = load_model('./model/model.h5')

@app.route('/model',methods = ['POST'])
def getImage():
    if request.method != 'POST':
        abort(404)
    else:
        file = request.files['image']
        image_stream = BytesIO()
        image_stream.write(file.read())
        image_stream.seek(0)
        file_bytes = np.asarray(bytearray(image_stream.read()),dtype=np.uint8)
        image_BGR = cv2.imdecode(file_bytes,cv2.IMREAD_COLOR)

        image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)

        image_RGB = convert_to_tensor(image_RGB)
        image_RGB = cast(image_RGB, float32)
        # print(image_RGB)
        image_RGB = (image_RGB / 127.5) - 1

        image = np.expand_dims(image_RGB,axis=0)

        images = []

        for i in range(0,4):
            generated_image = model(image, training=True)
            # generated_image = (generated_image * 0.5) + 0.5
            generated_image = (generated_image * 127.5) + 127.5

            # print(generated_image)

            output_RGB = np.squeeze(generated_image,axis=0)
            output_BGR = cv2.cvtColor(output_RGB,cv2.COLOR_RGB2BGR)

            # to save image on Server
            # cv2.imwrite('something.png',ans)

            status, buffer = cv2.imencode('.png',output_BGR)
            if status:
                # images.append(base64.b64encode(buffer.tobytes()))
                images.append(base64.b64encode(buffer.tobytes()).decode('utf-8'))

        print(images)
        return jsonify(images)