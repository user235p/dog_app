import numpy as np
import csv
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dense
from keras.models import Sequential 
import keras.utils as image

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib

app = Flask(__name__)


def breed_prediction(img_path, model):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    img_tensor = np.expand_dims(x, axis=0)
    img_tensor_normalized = img_tensor.astype('float32')/255
    # Get index of predicted dog breed for each image in test set
    dog_breed_prediction = np.argmax(model.predict(img_tensor_normalized))
    dog_breed_name = breed_dict[str(dog_breed_prediction)]
    #print('The breed of the given picture is likely a dog breed "',dog_breed_name , '"')
    return dog_breed_name

# Load breed dictionary
with open('breed_dict.csv') as csv_file:
    reader = csv.reader(csv_file)
    breed_dict = dict(reader)
    
    
# Create CNN model
model = Sequential()
num_classes = len(breed_dict)

# Architecture definition.
input_size = [224, 224, 3] # Size of the pictures and 3 color layers.
# 1st layer convulotional is the imput layer with the size of the images
model.add(Conv2D(filters=16,kernel_size=2, strides=1, activation='relu', input_shape= input_size))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2, strides=1,activation='relu', input_shape= input_size))
model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
model.add(Conv2D(filters=64,kernel_size=2, strides=1,activation='relu', input_shape= input_size))
model.add(MaxPooling2D(pool_size=2))
model.add(GlobalAveragePooling2D())
model.add(Dense(num_classes, activation='softmax')) # we use activation softmax for clasification problem.

# compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#Load the Model with the Best Validation Loss
model.load_weights('weights.best.from_scratch.hdf5')

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # render web page with plotly graphs
    return render_template('master.html') #, ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    file_path = "../app_images/"+query

    # use model to predict classification for query
    prediction_label = breed_prediction(file_path, model)

    #print (prediction_label)

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=prediction_label

    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()