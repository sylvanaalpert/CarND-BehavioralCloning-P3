# **Behavioral Cloning Project**

## By Sylvana Alpert
---

This project used a driving simulator to collect data demonstrating good driving behavior. Those images were then fed to a deep convolutional neural network to predict steering angles. The model was validated with a validation set and tested on the simulator in autonomous driving mode. The model was able to successfully drive around the track without leaving the confines of the road and with no need of any user intervention.

[//]: # (Image References)

[image1]: ./examples/centerLane.jpg "Center Lane Driving"
[image2]: ./examples/recoveryLeft.jpg "Recovery Image From the Left"
[image3]: ./examples/recoveryRight.jpg "Recovery Image From the Right"
[image4]: ./examples/original.jpg "Original"
[image5]: ./examples/flipped.jpg "Flipped"

---
### Submitted Files

My project includes the following files:
* clone.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode (unmodified code, as provided by Udacity)
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results
* autonomous_driving.mp4 demostrating how the model drives the car around the car autonomously


Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

The clone.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. Please note that if a file named model.h5 is present in the same directory as this file, that model will be used as a starting point for the training.

### Model Architecture and Training Strategy

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 256 (clone.py lines 77-97). There are four identical convolutional blocks (which differ mainly in the number of filters), each containing two convolutional layers with RELU activation and a max pooling operation to downsample the data.

The input is cropped, spatially downsampled and normalized using Keras layers Cropping2D, AveragePooling2D and Lambda, respectively (code lines 72-74).

The model contains one dropout layer in order to reduce overfitting. This layer is placed after the Fully Connected layer which is used before the output layer (clone.py line 105).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (clone.py line 117).

#### Training data

The sample training data, provided by Udacity, was used to train and validate the model. Additional data was collected for the following situations: (a) driving in the opposite direction of the track to prevent a left turning bias and (b) recovery driving from the sides of the track. However, these additional images were not used to train the model as they worsened the results of the model. This was presumably caused by poor steering angle resolution in the data I collected (without using a joystick).

#### Solution Design Approach

The overall strategy for deriving a model architecture was to start from a VGG16-like architecture. In each convolutional block, VGG16 contains three convolutional layers. Here, I used only two convolutional layers, to reduce the memory requirements and allow a faster training. This architecture seemed appropriate because of the large number of convolutional layers with non-linearities, which would allow the model to learn the image properties that would correlate with the appropriate steering angles.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. The mean square error was higher on the validation set, which implied that the model was overfitting. Indeed, when tested on autonomous mode, this model did not perform well on the track.

To combat this, I decided to add data augmentations to allow the model to generalize better to right and left turns and add a dropout layer before the output layer.

The data augmentations included left-right flipping of the center images and using the left and right camera images as well. This increased the training set size by a factor of 4.

This new model had both low train a validation errors and when tested in the simulator, the vehicle was able to drive autonomously around the track without leaving the road.

#### Final Model Architecture

The final model architecture (clone.py lines 69-107) consisted of a convolution neural network with the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 160x320x3 RGB image   							|
| Cropping         		| 50 rows from the top of the image and 20 from the bottom. Output size: 90x320x3   							|
| Downsample         		| Output size: 45x160x3   							|
| Normalization         		| Output image in range 0.5 to -0.5   							|
| Convolution 3x3     	| 32 filters, 1x1 stride, same padding, RELU	|
| Convolution 3x3     	| 32 filters, 1x1 stride, same padding, RELU 	|
| Max pooling	      	| 2x2 size, 2x2 stride 		|
| Convolution 3x3     	| 64 filters, 1x1 stride, same padding, RELU	|
| Convolution 3x3     	| 64 filters, 1x1 stride, same padding, RELU 	|
| Max pooling	      	| 2x2 size, 2x2 stride 		|
| Convolution 3x3     	| 128 filters, 1x1 stride, same padding, RELU	|
| Convolution 3x3     	| 128 filters, 1x1 stride, same padding, RELU 	|
| Max pooling	      	| 2x2 size, 2x2 stride 		|
| Convolution 3x3     	| 256 filters, 1x1 stride, same padding, RELU	|
| Convolution 3x3     	| 256 filters, 1x1 stride, same padding, RELU 	|
| Max pooling	      	| 2x2 size, 2x2 stride 		|
| Fully connected		| Outputs 4096x1								|
| Dropout 					|												|
| RELU					|												|
| Fully connected		| Outputs 1x1								|


#### Creation of the Training Set & Training Process

The sample data set provided with the project guidelines was used to train the model on good driving behavior. This data set contains many examples of center lane driving. Here is an example image of center lane driving:

![Center lane driving][image1]

I recorded the vehicle recovering from the left side and right sides of the road back to center so that the model would learn to go back to the center in case the vehicle got too close to the edges. These images show what a recovery looks like starting from the left and right respectively:

![Recovery Driving Left][image2]
![Recovery Driving Right][image3]

In addition, the vehicle was recorded when driving in the opposite direction of the track to prevent any bias on the turning direction.

Ultimately, these data samples were not used to train the model as the sample data by itself yielded better results overall.

I randomly split the data set into training and validation sets, with the latter containing 20% of the samples.

To augment the data set, I also flipped images horizontally and and used the two side camera images provided by the simulator. Flipping the images helped prevent any bias towards the left (the direction that the track loops around). Using the side cameras with a minor correction to the steering angle, helped provide more data points for the model to learn how to behave.
Here is an example of a flipped image:

![Original][image4]
![Flipped][image5]


After the augmenting the data set, I had 25712 training data points.

The data was preprocessed by cropping irrelevant parts of the image, downsampling with a factor of 2, and normalizing each image to the range of [-0.5, 0.5].

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 as evidenced by the cease of change in loss in the validation set. I used an Adam optimizer so that manually training the learning rate wasn't necessary.
