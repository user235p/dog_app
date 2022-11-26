# Data Scientist Capstone: CNN for dog identification app

## Installations

The code is written in Python.
You need some libraries. For the application following libraries are necessary:
- Numpy
- csv
- Flask
- Keras

## Project Motivation

With this project, it is possible to identify on dog pictures witch dog breed is.

## File Descriptions

The code to train the model is at the following notebook file:
**dog_app.ipynb**

The flask application is under app/run.py

## How to Interact with the app

The model is already trained at the notebook and the best trained parameters are under **app/weights.best.from_scratch.hdf5**

Afterwards the app uses CNN trained model to make the dog breed classification.

In order to lunch the flask application, go into the app directory and execute python run.py
	
Once the web interface is launched, please give a file name that is available in app_images directory. The trained CNN model will classify the given image and give the most likely dob breed associated with it.	


## Licensing, Authors, Acknowledgements

I used the template from Udacity to complete the code.

Therefore here the license from Udacity:

Copyright (c) 2017 Udacity, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


