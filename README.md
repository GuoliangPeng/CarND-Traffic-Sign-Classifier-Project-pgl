## Project: Build a Traffic Sign Recognition Program
## Deep Learning
## Result:
本程序实现一个可以识别交通标志的卷积神经网络，输入为一张32 * 32的彩色图像，输出为43种交通标志的预测结果。
![](https://github.com/GuoliangPeng/CarND-Traffic-Sign-Classifier-Project-pgl/blob/master/image6.jpg)

Overview
---
In this project, you
will use what you've learned about deep neural networks and convolutional neural
networks to classify traffic signs. You will train and validate a model so it
can classify traffic sign images using the [German Traffic Sign
Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After
the model is trained, you will then try out your model on images of German
traffic signs that you find on the web.

The project code and description are contained in Traffic_Sign_Classifie.ipynb.

The Project
---
The goals / steps of this project are
the following:
* Load the data set
* Explore, summarize and visualize the data
set
* Design, train and test a model architecture
* Use the model to make
predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./image1.jpg "Sample distribution
within different data sets"
[image2]: ./image2.jpg "Sample distribution of train
data sets"
[image3]: ./image3.jpg "Train and Validation Accuracy "
[image4]:./image4.jpg "Traffic Sign 1"
[image5]: ./image5.jpg "Traffic Sign 2"
[image6]: ./imgae6.jpg "Traffic Sign 3"


### Dependencies
This lab
requires:

* [CarND Term1 Starter Kit](https://github.com/GuoliangPeng/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter
Kit. Click [here](https://github.com/GuoliangPeng/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1.Download the data set. This is a pickled dataset in which we've already resized
the images to 32x32. It contains a training, validation and test set.  
2. Clone the project, which contains the Ipython notebook.

```{.python .input .sh}
git clone https://github.com/GuoliangPeng/CarND-Traffic-Sign-Classifier-Project-pgl.git
cd CarND-Traffic-Sign-Classifier-Project-pgl
jupyter notebook Traffic_Sign_Classifier.ipynb
```


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. 
In the code, the
analysis should be done using python, numpy and/or pandas methods rather than
hardcoding results manually.

I used the python, matplotlib to calculate summary
statistics of the traffic

* The size of training set is 34799
* The size of the
validation set is 4410
* The size of test set is 12630
* The shape of a traffic
sign image is (32, 32, 3)
* The number of unique classes/labels in the data set
is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an
exploratory visualization of the data set. 
It is a bar chart showing sample
distribution within different data sets.

![alt text][image1]

It is a bar chart
showing sample distribution of train data sets.
![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how I preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Minimally, the image data should be normalized so that the
data has mean zero and equal variance. For image data, (pixel - 128)/ 128 is a
quick way to approximately normalize the data and can be used in this project.
#### 2. Describe what my final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the
following layers:

| Layer         		|     Description	        					        |
|:---------------------:|:-----------------------------------------------------:|
| Input         		| 32x32x3 RGB image   							        | 
| Convolution1  5x5
| 1x1 stride, VALID padding, outputs 28x28x12           |
| RELU					|
|
| Dropout				|												        |
| Convolution2  3X3     | 1x1 stride,
VALID padding, outputs 26x26x24           |
| RELU					|
|
| Dropout				|												        |
| Max pooling	      	| 2x2 size, 2x2
stride, VALID padding, outputs 13x13x24	|
| Convolution3  3x3	    | 1x1 stride,
VALID padding, outputs 11x11x48           |
| RELU					|												        |
|
Dropout				|												        |
| Max pooling	      	| 2x2 size, 2x2 stride,
VALID padding, outputs 5x5x48	|
| Flatten       		| outputs 1200
|
| Fully connected		| outputs 500       									|
| RELU					|
|
| Fully connected		| outputs 84        									|
| RELU					|
|
| Fully connected		| outputs 43        									|
| Logits				| outputs 43
|
 

#### 3. Describe how I trained my model. 
为了训练模型，我使用了AdamOptimizer优化器，批量大小为128,epochs=25,learningrate=0.0007,
开始的时候使用了0.003的学习率，，训练20次，模型的验证集准确度波动较大，所以我将学习率减小，并
设定训练集的dropout=0.83

#### 4.Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. 
Include in the discussion the results on the
training, validation and test sets and where in the code these were calculated.
My approach may have been an iterative process, in which case, outline the
steps
I took to get to the final solution and why you chose those steps.
Perhaps my
solution involved an already well known implementation or
architecture. In
this
case, discuss why I think the architecture is suitable
for the current
problem.
My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.963 
* test set accuracy of 0.949

If an iterative approach was chosen:
* 我首先尝试了LeNet架构，验证集的精度低于0.93，意味着需要对模型进行修改
* 我尝试在卷积层添加dropout层设为0.9，并调整学习率从0.001到0.003,防止权重过拟合， 验证集能达到0.93-0.94左右的准确率，
* 可能是卷积层太少导致深层次特征未能准确提取，于是添加了一层卷积层，调整了滤波器大小，
发现训练集和验证集结构波动很大，所以调整学习率为0.0007，并设dropout为0.83，防止过拟合，提高模型泛化能力
* 卷积在图像识别领域工作得很好，我们的项目在这个领域，所以我们可以使用卷积层来解决这个问题。dropout层可以有效地避免过度拟合
这是最终的模型学习曲线，从曲线我们可以看出列车和有效精度都收敛到一个很高的值，它们之间的差距很小，所以我认为模型非常合适，
没有过度拟合或不合适。从最后两个时代开始，出现了过度拟合的趋势，所以我们不应该设置太大的时代数，大约25个是合适的。

![alt text][image3]

### Test a Model on New Images

#### 1.Choose ten German traffic signs found on
the web and provide them in the report. For each image, discuss what quality or
qualities might be difficult to classify.

Here are ten German traffic signs
that I found on the web:

![alt
text][image4]

第3,4,5,6,7个图像是比较模糊且色彩偏暗，比较难以识别。
#### 2. Discuss the model's predictions on these new traffic signs and compare
the results to predicting on the test set. At a minimum, discuss what the
predictions were, the accuracy on these new predictions, and compare the
accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more
detail as described in the "Stand Out Suggestions" part of the rubric).

Here
are the results of the prediction:

![alt text][image5]


The model was able to
correctly
guess 9 of the 10 traffic signs, which gives an accuracy of 90%.
#### 3. Describe how certain the model is when predicting on each of the ten new
images by looking at the softmax probabilities for each prediction. Provide the
top 5 softmax probabilities for each image along with the sign type of each
probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the
rubric, visualizations can also be provided such as bar charts)

The code for
making
predictions on my final model is located in the cell of the Ipython
notebook.

![alt text][image6]


### (Optional)
Visualizing the Neural Network
(See
Step 4 of the Ipython notebook for more
details)
#### 1. Discuss the visual
output of your trained network's feature
maps. What characteristics did the
neural network use to make classifications?

