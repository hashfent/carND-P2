
# **Traffic Sign Recognition** 

## Writeup 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set 
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./data/histo.jpg "Histogram"
[image2]: ./real_test_images/0.jpg "Test image 1"
[image3]: ./real_test_images/1.jpg "Test image 2"
[image4]: ./real_test_images/2.jpg "Test image 3"
[image5]: ./real_test_images/3.jpg "Test image 4"
[image6]: ./real_test_images/4.jpg "Test image 5"
[image7]: ./data/real_vs_train.jpg "Web image vs predicted"


## Summary of the dataset
The dataset used in this project is images of German Traffic Signs  


## Rubric Points
* The training, validation and test datasets have been pre-processed. 
* I have been able to achieve a validation accuracy of over 93%.
* I have included a write-up summary report.

---
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training data is distributed. 

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Only normalization of the data with 0 mean has been applied to all the datasets. No transformation to grayscale was applied as I judged color data could 
help in the classification of the images.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 32x32x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU 					| 	        									|
| Max pooling 			| 2x2 stride,  outputs 5x5x16      				|
| Fully connected		| Input = 400, Output = 120        				|
| RELU					| 	       										|
| Fully connected		| Input = 120, Output = 84						|
| RELU					| 	       										|
| Fully connected		| Input = 84, Output = 10						|
| Softmax				| returns logits								|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a batch size of 128 and trained it for 50 epochs using the pre-processed training dataset. A learning rate of 0.001 was applied.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The training accuracy was quickly converging to a reasonable value. However the validation accuracy was not easily met by the standard LeNet network which was chosen. The reason was because of over fitting as well as the validation dataset being a small one.
In order to get a validation accuracy that meets the requirment I included an L2 loss regularization factor (2e-4) as well as a dropout witha factor of 70%.


My final model results were:
* training set accuracy of 99.1%
* validation set accuracy of 93.4%
* test set accuracy of 92.3%


If a well known architecture was chosen:
* The LeNet architecture was chosen for this project.
* It was chosen because,as a CNN based architecture, it is well suited for image classification.
* The training accuracy is quite high at 99.1%, a 94% validation accuracy and a 93% test accuracy are reasonable. This suggests that the chosen architecture was appropriate for this task.
 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:
These images are screen shots of Google's street map as I navigated around German towns.

![alt text][image2] ![alt text][image3] ![alt text][image4] 
![alt text][image5] ![alt text][image6]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road work     		| Road work   									| 
| Speed limit (30km/h)  | Speed limit (60km/h)       					|
| No passing			| No passing									|
| Yield	                | Yield			        		 				|
| No entry			    | No entry           							|

The image below shows the comparison between web downloaded test images and the classification of the corresponding images by the network.
![alt_text][image7]

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. Compared to the accuracy on the test set of 92.3% this is not quite high. However this can be expected because of the low number of real test images which is only 5.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


For the fourth image, the model is relatively sure that this is a stop sign (probability of 0.4), and the image does contain a stop sign. However for the second image the probablity is only 0.05 and the model did not predict it correctly. However, even a lower probablity of 0.02 for the fifth image does predict the right sign. This suggests that it would be difficult to rely on on probablity values as a sign of confidence for such a small set of test images. 

The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .25         			| Road work   									| 
| .05     				| Speed limit (60km/h) 							|
| .22					| No passing									|
| .40	      			| Yield			       			 				|
| .02				    | No entry            							|



```python

```
