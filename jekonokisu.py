from flask import Flask,render_template,request,session,logging,url_for,redirect
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session,sessionmaker
from passlib.hash import sha256_crypt

from flask import render_template
from flask import Response

import cv2

# pip install imutils
from imutils.video import VideoStream
import imutils

app = Flask(__name__)
videoStream = VideoStream(src=0).start()#for mobile cam
videoStream1 = VideoStream(src=1).start() #web or laptop.

import numpy as np
import cv2



@app.route("/")
def index():
    return render_template("stream.html")






@app.route("/video")
def video():
    return Response(generateFrames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generateFrames():
    while True:
        frame = videoStream.read()
        frame = imutils.resize(frame, width=600)
        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n') 




# ------------------------------------------Second Camera------------------------------------------------
@app.route("/video1")
def video1():
    return Response(generateFrames1(), mimetype='multipart/x-mixed-replace; boundary=frame')
def generateFrames1():
    while True:
     frame1 = videoStream1.read()
     frame1 = imutils.resize(frame1, width=800)
     (flag, encodedImage) = cv2.imencode(".jpg", frame1)
     yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')


       
        

if __name__ == '__main__':
    app.run(debug=True)