import cv2
import numpy as np
import base64
from flask import Flask, json, request, jsonify, Response,make_response
from flask_cors import CORS
from ocr import *
from detection import *

app = Flask(__name__)
CORS(app)
app.config['DEBUG'] = True


@app.route('/api/detection', methods=['GET','POST'])
def detect():
    try:
        imageb = request.get_json()['image']
    except:
         return jsonify({'message':'no image sent!'})
    try:
        with open("getImage.png", "wb") as fh:
            fh.write(base64.b64decode(str(imageb)))
        
    except:
        return jsonify({'message':'error image format!'})
    try:
        image=cv2.imread('getImage.png')
        detected_image = detection(image)
        index,scores=ocr(detected_image)
        if not index:
            return jsonify({'message':'ocr nonfunctional'})
        cv2.imwrite('postImage.png',detected_image)
        with open('postImage.png', "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        return jsonify({'index':index,'scores':scores}) ##'detection':str(encoded_string,'utf-8')
    except:
        return jsonify({'message':'ocr or detection error'})



if __name__ == "__main__":
    app.run()


