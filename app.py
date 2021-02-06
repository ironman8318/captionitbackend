from flask import Flask,jsonify,request
from flask_mysql_connector import MySQL

# from flask_mysqldb import MySQL
app = Flask(__name__)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DATABASE'] = 'image_caption'

mysql = MySQL(app)


import numpy as np
import urllib
import cv2

def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	# return the image
	return image


@app.route('/',methods=['GET'])
def hello():
    imgURL = "https://s3.ap-south-1.amazonaws.com/gocomet-images/carriers/logo/one-line.png"
    
    # Pass imageURL into url_to_image function
    image = url_to_image(imgURL)
    # caption = classifier.predict(np.array(image).reshape(1,-1))
    caption = "This is beautiful"
    
    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO caption(image, caption) VALUES (%s, %s)", (imgURL, caption))
    mysql.connection.commit()
    cur.close()
    return "Done"

@app.route('/search',methods=['POST'])
def search():
    _json = (request.json)
    caption = _json['content']
    
    try:
        conn = mysql.connection
        cur = conn.cursor()
        # caption = "'%" + caption + "%'"
        query = "SELECT image FROM caption WHERE img_caption LIKE %s"
        value = (caption, )
        cur.execute(query,value)
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


