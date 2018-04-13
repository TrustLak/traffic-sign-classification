
# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/TrustLak/traffic-sign-classification)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Sample of the dataset is visualized in the IPython notebook



### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

As a first step, I decided to convert the images to grayscale because I believe there is little information in color. In some cases however, the red or blue color can help make better predictions. We focus however on the grayscale.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because this will make the optimization process much faster. In may cases, this step is necessary for the convergence of the optimization algorithm.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My model is a slight variation of LeNet architecture. Some of the filters have different dimensions. The details are included in the following table:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 (grayscale)   							| 
| Convolution 3x3     	| filter_hight = 5, filter_width = 5, in_depth = 6, out_depth = 20, stride = 1x1, padding = 'VALID' 	|
| tanh					|												|
| Max pooling	      	| filter_hight = 2, filter_width = 2, stride = 2x2, padding = 'VALID' 				|
| Convolution 3x3	    | filter_hight = 5, filter_width = 5, in_depth = 1, out_depth = 6, stride = 1x1, padding = 'VALID' |
| tanh					|												|
| Max pooling	      	| filter_hight = 2, filter_width = 2, stride = 2x2, padding = 'VALID' 				|
| flatten					|												|
| Fully connected		| input: 500x1, output: 200       									|
| tanh					|												|
| Fully connected		| input: 200x1, output: 100       									|
|	sigmoid					|												|
| Fully connected		| input: 100x1, output: 43       									|
 
The last layers is our logits, which are passed to a softmax layer. The softmax output is fed to a one-hot functions, which gives us our predictions.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

| Variable         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Optimizer		| Adam       			|  
| Loss function		| Cross entropy       			|  
| Batch size		| 128      			|  
| Learning rate		|  0.0015       			|  
| Epochs		|  10       			|  
| Layer initialization		|  Truncated normals with mu = 0 and sigma = 0.1  			|  


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.954 
* test set accuracy of 0.931

If an iterative approach was chosen:
* I start off with the exact LeNet architecture with modified dimensions of the fully connected layers (see details of fully connected layers above). After varying the learning rate and batch size, the best validation accuracy was 0.89.
* Noticing that some classes of the dataset are under-represented, I removed dropout layers. The validation accuracy slightly to 0.91.
* I decided to try grayscale, to reduce the size of the network (in order to avoid overfitting). The accuracy slightly increases. So I decided to stick with grayscale.
* After varying the learning rate and batch size, the validation accuracy was still below 0.93. My initial impelentation of LeNet used sigmoid activation functions. I tried different activations such as tanh and relu to make sure that I don't have vanishing gradient problems. I found its best to use tanh activations, with the exception of the last one (set to signmoid).
* The results were satisfactory, i.e. slightly above 0.93 accuracy. 
* One more additional change inluded increasing the depth of the first two convolutional layers. This change increased the validation accruacy to 0.96.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text](60_kmh.jpg) ![alt text](road_work.jpg) ![alt text](stop_sign.jpg) 
![alt text](yield_sign.jpg) ![alt text](left_turn.jpeg)

The first image might be difficult to classify because of the noise around the traffic sign (image not properly zoomed on the traffic sign).

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|  Confidence           | Correct prediction among top 5? |
|:---------------------:|:-----------------------:|:--------------------------------:|:--------------------------------:| 
| 60_kmh      		| No passing   									| medium   									| Yes 									| 
| road_work     			| General caution    	| low   									| No  									|
| stop				| stop  											| high   									| Yes		|
| yield	      		| yield					 					| high   									| Yes  									|
| left_turn			| Slippery Road      								| low   									| Yes  									|

The exact probabilites are detailed in the IPython notebook.

The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This does not compare well to the test accuracy. This was expected since many classes of our dataset have less than 500 images, which means there not enough training data that allows the network to generalize(dataset not rich enough). Also, the noise in these new images around the traffic signs is relatively higher than that in the training data. One way to help remove the effect of the noise is to include RGB layers in the training, rather than just grayscale (which is opposite to my initial intuition while designing the network. This will be added to future work).






