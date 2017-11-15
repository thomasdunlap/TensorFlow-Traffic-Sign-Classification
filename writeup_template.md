# **Traffic Sign Recognition**

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

[bar_train]: ./examples/barplot_training.png "Visualization"
[stop_sign]: ./new_images/stop_sign.jpg "Stop Sign"
[bar_post_rot]: ./examples/barplot_after_rot.png "Bar plot of augmented data set"
[30km_sign]: ./new_images/30km_sign.JPG "30 km/h Sign"
[70km_sign]: ./new_images/70km_sign.jpg "70 km/h Sign"
[no_entry_sign]: ./new_images/Do-Not-Enter_Sign.jpg "No Entry Sign"
[children_crossing]: ./new_images/children-crossing.jpg "Children Crossing Sign"
[right_turn_ahead]: ./new_images/right_turn_ahead.jpg "Right Turn Ahead Sign"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### README

#### Provide a README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/thomasdunlap/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy, and pandas methods rather than hardcoding results manually.

I used the pandas, numpy, and python to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### Visualization of the Dataset.

I used a bar chart to visualize the data set.

![alt text][image1]

Of the 43 unique labels, there is a wide variance in the number of examples in each category. I also visualized a random image from the training set, like this one:

![random image][image2] <= put real image.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

If I was using a CPU I'd put Batch size lower, but working with AWS GPU I moved it up to 256.  A batch size of 256 allows for a more general


I normalized the image data to 32 bit floating point images, because I wanted to eliminate the unnecessary variable of having the type of image influence the algorithm's learning process.

I decided to generate additional data because there was such great variance in the number of examples per label.  Labels with a small number of samples would be at high risk of overfitting.

To help account for the greatly varying number of examples in each dataset,

Here is an example of an original image and an augmented image:

![alt text][image2]  ![alt text][image3]

The augmented dataset was rotated randomly between -20 and 20 degrees.  This is to ensure that the signs are different, but not too different from the originals.  These angles were generated based on a zero-centered Gaussian distribution, with a standard deviation of 10. Also, any angle that would have been more than a 20 degree change from the original was a assigned to 15 degrees (or -20 degrees if too negative).  This would a assure the majority of the augmented images would have smaller rotations, centered around zero.  

I came to the conclusion of 20 degrees by trial an error.  20 was about the most rotation I found I could do before losing accuracy. I wanted to rotate the images as much as I could get away with to account for the varying angles a sign could be presented at.


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

* Learing rate: .00033
* Dropout/Keep Probability: .76
* Epochs: 80
* Batch size: 128

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ?
* test set accuracy of ?

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

![30km_sign][30km_sign]

![70km_sign][70km_sign]

![alt text][right_turn_ahead]

![alt text][no_entry_sign]

![alt text][stop_sign]

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


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .60         			| 30 km/h Sign   									|
| .20     				|  										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ...

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .60         			| 30 km/h Sign   									|
| .20     				|  										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the third image ...

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .60         			| 30 km/h Sign   									|
| .20     				|  										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the fourth image ...

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .60         			| 30 km/h Sign   									|
| .20     				|  										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the fifth image ...

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .60         			| 30 km/h Sign   									|
| .20     				|  										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


Inititally I thought about adding noise to the image, but then I realized some of the normalized images already had noise, and felt like it was overkill.
