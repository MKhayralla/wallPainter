from flask import Flask , render_template
from flask_cors import CORS
from flask_socketio import SocketIO , emit
import numpy as np
import cv2 as cv

#start of the algorithm : 
def process_img(image , x , y , r , g , b , cols , rows):
    pixels = []
    for i in range(cols*rows*3):
        pixels.append(image[str(i)])
    img = np.array(pixels).reshape(cols,rows,3).astype(np.uint8)
    #image segmentation :
#gray scale
    im_gr = cv.subtract(cv.cvtColor(img , cv.COLOR_BGR2GRAY),1)
#detecting edges to yse them as a mask in the flood fill step
    edges = cv.Canny(im_gr, 25 , 60 ,L2gradient= True )
#adjusting mask size
    mask = np.zeros((cols+2, rows+2), np.uint8)
    for i in range(cols):
        for j in range(rows):
            mask[i][j] = edges[i][j]
#looking for the wall connected component around the seed point 
    cv.floodFill(im_gr , mask , (x,y) , 255 , 5 ,5 )
#creating the alpha blending mask
    _ , alpha = cv.threshold(im_gr , 254 , 255 , cv.THRESH_BINARY)
#mask dilation to fill the gaps and remove pepper noise 
    alpha = cv.dilate(alpha , np.ones((5,5) , np.uint8))
    
    #alpha blending starts here : 
    alpha = cv.divide(alpha,255)

    result = np.zeros((cols,rows,3) , dtype = np.uint8)
    foreground = result 
    foreground[:,:,:] = [b,g,r]
    for i in range(cols):
        for j in range(rows):
            if alpha[i,j] == 0:
                result[i,j,0] = img[i,j,0]
                result[i,j,1] = img[i,j,1]
                result[i,j,2] = img[i,j,2]
            else:
                result[i,j,:] = foreground[i,j,:]*0.75 + img[i,j,:]*0.25
    return  result


app = Flask(__name__, static_url_path='')
CORS(app)
socketio = SocketIO(app)

@socketio.on('try')
@socketio.on('connected')
def handle_my_custom_event(json):
    print('received json: ' + str(json))
#handling the received message of the client : 
@socketio.on('proc')
def handle_proc(json):
    image = process_img(json['img'] , json['x'] , json['y'] , json['r'] ,json['g'] , json['b'] , json['cols'] , json['rows'])
    cv.imwrite('output.jpg',image)
    emit('img',{'link':'output.jpg'})
if __name__ == '__main__':
    socketio.run(app)
