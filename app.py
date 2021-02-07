import cv2
import urllib
import numpy as np
import urllib.request
import tensorflow as tf
from flask import Flask, jsonify, request
from pickle import load
from flask import Flask, request, jsonify, render_template
from img_caption import *

from flask import Flask, jsonify, request
from flask_mysql_connector import MySQL
from flask_cors import CORS, cross_origin

# from flask_mysqldb import MySQL
app = Flask(__name__)
CORS(app)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DATABASE'] = 'hackathon'

mysql = MySQL(app)


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    filename = url.split("/")[-1]
    res = urllib.request.urlretrieve(url,filename)
    # resp = urllib.request.urlopen(url)
    # image = np.asarray(bytearray(resp.read()), dtype="uint8")
    # image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # return the image
    
    
    return filename

def evaluate(image, tokenizer):

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result
        dec_input = tf.expand_dims([predicted_id], 0)

    
    return result



@cross_origin()
@app.route('/v1/predict', methods=['POST'])
def predict():
    # imgURL = "https://s3.ap-south-1.amazonaws.com/gocomet-images/carriers/logo/one-line.png"
    _json = (request.json)
    imgURL = _json['imgURL']

    # Pass imageURL into url_to_image function
    image = url_to_image(imgURL)
    checkpoint_path = "checkpoint/ckpt-4"
    train_captions = load(open('caption/captions.pkl', 'rb'))


    # Choose the top 5000 words from the vocabulary
    top_k = 5000
    train_seqs , tokenizer = tokenize_caption(top_k ,train_captions)

    #restoring the model

    ckpt = tf.train.Checkpoint(encoder=encoder,decoder=decoder,optimizer = optimizer)
    ckpt.restore(checkpoint_path)

    # to_predict_list = request.form.to_dict()
    # Image_path = to_predict_list['pic_url']

    # new_img =  Image_path

    result = evaluate(image, tokenizer)
    for i in result:
        if i=="<unk>":
            result.remove(i)
        else:
            pass

    caption = ' '.join(result).rsplit(' ', 1)[0]


    # caption = classifier.predict(np.array(image).reshape(1,-1))
    cur = mysql.connection.cursor()
    cur.execute(
        "INSERT INTO caption(image, caption) VALUES (%s, %s)", (imgURL, caption))
    mysql.connection.commit()
    cur.close()
    resp = jsonify(
        caption=caption
    )
    resp.status_code = 200
    return resp


@cross_origin()
@app.route('/v1/search', methods=['POST'])
def search():
    _json = (request.json)
    caption = _json['caption']

    try:
        conn = mysql.connection
        cur = conn.cursor()
        # caption = "'%" + caption + "%'"
        query = "SELECT image,caption FROM caption WHERE caption LIKE %s"
        value = (caption, )
        cur.execute(query, value)
        rows = cur.fetchall()
        resp = jsonify(rows)
        resp.status_code = 200
        return resp
    except Exception as e:
        print(e)
    finally:
        cur.close()


if __name__ == '__main__':
    app.run()
