#gunicorn --bind=0.0.0.0:8000 --workers=1 server:app

import os
from glob import glob

from flask import Flask, render_template, request, send_file
import caffe
from PIL import Image
import numpy as np

def preprocess_image(im):
    # preprocess image same way as for network
    # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    mean=np.array((104.00699, 116.66877, 122.67892))
    
    in_ = np.array(im, dtype=np.float32)
    
    if len(in_.shape) < 3:
        w, h = in_.shape
        ret = np.empty((w, h, 3), dtype=np.float32)
        ret[:, :, :] = in_[:, :, np.newaxis]
        in_ = ret
    
    # get rid of alpha dimension
    if in_.shape[2] == 4:
        background = Image.new("RGB", im.size, (255, 255, 255))
        background.paste(im, mask=im.split()[3]) # 3 is the alpha channel
        in_ = np.array(background, dtype=np.float32)
    
    in_ = in_[:,:,::-1]
    in_ -= mean
    in_ = in_.transpose((2,0,1))
    return in_

def calc_pred_importance(im_loc,net):
    im = Image.open(im_loc)
    in_ = preprocess_image(im)
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    net.forward()
    return net.blobs['loss'].data[0]

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

net = caffe.Net('./gdi/fcn16/deploy.prototxt','./models/gdi_fcn16.caffemodel', caffe.TEST) # CHANGETHIS

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return r

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        if not os.path.exists('./uploads'):
            os.makedirs('./uploads')

        files = glob('./uploads/*')
        for f in files:
            os.remove(f)

        file = request.files['image']
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

        file.save(filename)

        pred_imp = calc_pred_importance(filename, net)
        data = pred_imp[0,...]
        
        #Rescale to 0-255 and convert to uint8
        rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
        im_new = Image.fromarray(rescaled)

        new_loc = ''.join(filename.split('.')[:-1]) + '_new.png'
        im_new.save(new_loc,"PNG")

        return send_file(new_loc, attachment_filename=new_loc.split('/')[-1])

    else:
        return render_template('index.html')