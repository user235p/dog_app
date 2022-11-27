# Data Scientist Capstone: CNN for dog identification app

## Project Motivation

This project aims the goal to identify the dog breed based on a picture path provided by the user at a web hosted application.

## Methodology and metrics

The identification is based in a neural network witch is trained with dog images provided at the Udacity workspace. The data folder has 8351 dog labeled pictures, distributed in 133 dog breeds.

For such a purpose the neural network that fits good are usually based on convulotional layers (CNN).

At the notebook **dog_app.ipynb** the hole process is described.
There are two solutions. 

### CNN fron scratch

The first one is to start a neural network architecture from scratch. For this purpose a combination of convulotinal, pooling layers were checked. Afterwards a fully connected layer with the corresponding softmax activation function finalize the classification and deliver the probability between 0 and 1 for each dog breed.

The following architecture was implemented, trained and scored:

![CNN scratch architecture](file:///readme_images/CNN_scratch_architecture.png)   

In order to canculate the test accuracy, we sum the correct number of breed predictions through the number of predictions.

Accuracy = 100 * (number of right predictions / number of predictions)

In order to evaluate the breed, we take from the output of the dense layer, the position with the higher probability. For this purpose we use the numpy function np.argmax.

After training the model for 20 epochs on a train subgroup of the pictures and validating the training with a validation group, the accuracy of the model is calcuated based on predictions on a test group of pictures. This test group is totally separated from the train and validation groups.

For this architecture, the achived accuracy was 5.5%.

In order to improve the accuracy, we developed a seconds solution.

### Transfer learning model

For this purpose I used an already trained RestNet50 network, and added at the end a new pooling layer and a fully connected layer with so many nodes as dog breeds, using again the softmax activation function.

![Final layers](file:///readme_images/FinalLayersForRestNet50_architecture.png)   

The whole architecture of RestNet50 and these two final layers was retrained for 20 epochs.

Using the same metrics for accuracy, the model achived 78.8%.

## Installations and Working enviroment

The code is written in Python.
In order to run the project you need to create a python enviroment based on the library requirements documented under /requirements/requirements.txt

Here you have the procedure in Windows:

In order to create the enviroment, please execute the following:

__python -m venv env__

Activate the new enviroment with the following command:

__env/bin/activate.bat__

Then install the necessary libraries into your enviroment using the following command:

__pip install -r requirements/requirements.txt__

Now you are able to go throug the notebooks and execute the flask application.

## File Descriptions

The code to train the model is at the following notebook file:
**dog_app.ipynb**

The flask application is under **app/run.py**

In order to debug and prepare the **run.py** code, I used the **app_preparation.ipynb** notebook.

## How to Interact with the app

The model is already trained at the notebook and the best trained parameters are under **app/weights.best.from_scratch.hdf5**

Afterwards the app uses CNN trained model to make the dog breed classification.

In order to launch the flask application, go into the app directory and execute __python run.py__

The web interface will be lunched. Please pay attention to the showed http link in order to open it in your browser to interact with the app. Here you see where to find it.

![Web interface link](file:///readme_images/Weapp_address.png)    

Once the web interface is running at your browser, please give a file name that is available in app_images directory. The trained CNN model will classify the given image and give the most likely dob breed associated with it.	

![Web interface](file:///readme_images/Webapp_interface.png)    


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


