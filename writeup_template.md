# **Traffic Sign Recognition**
![Normalized Image][norm_38] ![Stop Sign][rand_stop] ![Rotated Sign][rotated_img]

---

**Build a Traffic Sign Recognition Project**

The goals of this project are:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[stop_bar]: ./examples/stop_bar.png "30km Bar Plot Softmax"
[rotated_img]: ./examples/rotated_img.png "30km Bar Plot Softmax"
[right_turn_bar]: ./examples/right_turn_bar.png "30km Bar Plot Softmax"
[30_bar]: ./examples/30km_bar.png "30km Bar Plot Softmax"
[70_bar]: ./examples/70km_bar.png "70km Bar Plot Softmax"
[no_entry_bar]: ./examples/no_entry_bar.png "No Entry Softmax"
[norm_38]: ./examples/normed_38_img.png "Normalized Image"
[rand_stop]: ./examples/random_stop.png "Random Stop Sign"
[bar_train]: ./examples/barplot_training.png "Visualization"
[bar_after_rot]: ./examples/bar_after_rot.png "Bar plot of data set after adding rotated images"
[random_img_38]: ./examples/rand_img.png "Children Crossing Sign"
[30km_sign]: ./new_images/30km_sign.JPG "30 km/h Sign"
[70km_sign]: ./new_images/70km_sign.jpg "70 km/h Sign"
[right_turn_ahead]: ./new_images/right_turn_ahead.jpg "Right Turn Ahead Sign"
[stop_sign]: ./new_images/stop_sign.jpg "Stop Sign"
[no_entry]: ./new_images/Do-Not-Enter_Sign.jpg "No Entry Sign"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### README

#### Provide a README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/thomasdunlap/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).

### Data Set Summary & Exploration

#### Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy, and pandas methods rather than hardcoding results manually.

I used the pandas, numpy, and python to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### Visualization of the Dataset.

I used a bar chart to visualize the data set:

![Bar Chart of Training Examples][bar_train]

There is a wide variance in number of examples across the 43 types of signs, ranging from 200 examples to 2000. I also visualized a random image from the training set, to get a feel for what the images looked like:

![Random image of German Stop Sign][rand_stop]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

#### Normalization

Data preprocessing was done in several steps. I normalized the image data to 32-bit floating point images:

```python
def norm32(X):
    """
    Normalize image and convert it to 32-bit floating point
    """
    norm = np.float32(128)
    return (X.astype(np.float32) - norm) / norm
```

This makes sure all the image data has a mean of zero and equal variance, and eliminates the possibility of having data type of image influence the algorithm's learning process.  Here is a random image before and after normalization:

![random image][random_img_38] ![normed image][norm_38]

#### Generating Additional Data

I decided to generate additional data because there was such great variance in the number of examples per label (anywhere from 200 to 2000).  Labels with a small number of samples would be at high risk of overfitting. To rotate images, I used created the following function:

```python
# To be used in rotation of images
import scipy.ndimage as scnd

def rotate_image(img):
    """
    Rotate image by random angle from Gaussian distribution with a stddev of 7, centered at zero.
    """

    # Create angle from random Gausian distribution with a stddev of 7, centered at zero
    angle = np.random.normal(0., 7.)

    # Rotates pixel array, then essentially houses it inside larger scale blank pixel array
    rotated_img = scnd.interpolation.rotate(img, angle)

    # Rotation creates larger image; crop to 32x32:
    margin = rotated_img.shape[0] - 32 # New rows minus original number of rows
    start_crop = np.floor(margin / 2).astype(int) # Divide by 2 because edge added both sides
    end_crop = start_crop + 32 # Will take 32 rows past bgn_zoomed
    cropped_rotation_img = rotated_img[start_crop:end_crop, start_crop:end_crop, :] # Crop image

    return cropped_rotation_img
```

This uses `scnd.interpolation.rotate(img, angle)` to rotate an image array a specificed number of degrees, and then essentiallyencases that rotated image inside the smallest black rectangle that will fit around it:  

![A rotated image.][rotated_img]

The new data was rotated based on an angle chosen randomly from a  Gaussian distribution centered at zero, with a standard deviation of 7 degrees.  This is to ensure that the majority of the additional images would be extremely similar to their originals, with a few larger rotations mixed in for variance and reducing the possibility of overfitting.

If I was using a CPU I'd put Batch size lower, but working with AWS GPU I moved it up to 256.  A batch size of 256 allows for a more general


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|	Activation											|
| Dropout       |  Keep Prob = .76      |
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16      									|
| RELU					|	Activation											|
| Dropout       |  Keep Prob = .76      |
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten       | outputs 400           |
| RELU					|		Activation										|
| Dropout       |  Keep Prob = .76      |
| Fully connected		| outputs 120       |
| RELU					|		Activation										|
| Dropout       |  Keep Prob = .76      |
| Fully connected		| outputs 84       |
| RELU					|	Activation											|
| Dropout       |  Keep Prob = .76      |
| Fully connected		| outputs 43       |
| Softmax				| outputs classification probabilities        									|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following hyperparameters:

* Learing rate: .000933
* Dropout/Keep Probability: .76
* Epochs: 80
* Batch size: 256

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy: 1.000
* validation set accuracy: .962
* test set accuracy: .951

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

I choose to use a slightly modified LeNet architecture. LeNet is an architecture I worked with in identifying handwritten numbers, and I thought it would translate well to identifying traffic signs.  The final model accuracy was !AN UNKNOWN NUMBER!, which suggests LeNet was a success.

As far as some of the hyperparameters go, I just sort guessed to low, then too high, until I reached a happy medium.  My hunch was since LeNet was trained to identify 10 numbers, and we were using it to identify 43 signs, the learning rate would need to be smaller and the number of epochs would need to be larger.  This made sense since there are more nuances to learn, and there are no real time constraints on how long it takes the model to train except my own patience.  This would hopefully help keep our Adam optimizer from getting stuck before it reached a minima.  As I'd observe the updates with each epoch, I noticed the optimizer getting "stuck" around certain numbers, and I figured if there was more room to descend down the gradient, the algorithm might be overshooting due to a too-large learning rate.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![30km/h sign][30km_sign]

![70km/h sign][70km_sign]

![Right Turn Ahead Sign][right_turn_ahead]

![No Entry Sign][no_entry]

![Stop Sign][stop_sign]

My model had an easy time classifying NUMBER of the five signs.  To be fair, they were all cropped to look like the original images.

Images where the sign was rotated significantly or off-center were difficult for the model to classify.  Which makes sense, seeing as I did not train the model with those types of images.  

In the future, it would probably be helpful to have larger rotations and more obscured signs.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 30 km/h      		| 30 km/h   									|
| 70 km/h    			| 70 km/h 										|
| Right Turn Ahead			| Right Turn Ahead											|
| No Entry	      		| No Entry				 				|
| Stop Sign			| Stop Sign     							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.












For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .99770          			| 30 km/h   									|
| .00120     				| 20 km/h 										|
| .00053					| 60 km/h											|
| .00050	      			| 50 km/h					 				|
| .00003				    | 120 km/h      							|

![30 km/h softmax bar plot][30_bar]

For the second image ...

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .63859         			| 70 km/h  									|
| .35192     				|  30 km/h										|
| .00528					| 50 km/h											|
| .00156	      			| 80 km/h					 				|
| .00090				    | 60 km/h      							|

![70 km/h softmax bar plot][70_bar]

For the third image ...

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			| Right Turn Ahead  									|
| 0     				| Roundabout Mandatory  										|
| 0					| No vehicles over 3.5 metric tons									|
| 0	      			|  70 km/h					 				|
| 0				    | Keep Left      							|

![Right turn softmax bar plot][right_turn_bar]

For the fourth image ...

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			| No Entry   									|
| 0     				|  No Passing									|
| 0					| Stop Sign											|
| 0	      			| No passing for vehicles over 3.5 tons				 				|
| 0				    | Dangerous curve to the right     							|

![No Entry softmax bar plot][no_entry_bar]

For the fifth image ...

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .99999         			| Stop Sign  									|
| .00000     				| Yield 										|
| .00000					| 20 km/h									|
| .00000	      			| 30 km/h					 				|
| .00000				    | No passing for vehicles over 3.5 tons     							|

![Stop sign softmax bar plot][stop_bar]

Inititally I thought about adding noise to the image, but then I realized some of the normalized images already had noise, and felt like it was overkill.

The annoying thing about drawing from a random distribution is when you try to recreate it and its different each time! I guess that's what random seed is for, but holy mother swearing!!!
