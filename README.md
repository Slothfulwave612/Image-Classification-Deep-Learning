# Image-Classification-Deep-Learning

## Overview:

* **Deep Learning**: A subset of Machine Learning Algorithms that is very good at recognizing patterns but typically requires a large number of data.

* **Neural Network**: A computational model that works in a similar way to the neurons in the human brain. Each neuron takes an input, performs some operations then passes the output to the following neuron.

  ![nn](https://user-images.githubusercontent.com/33928040/75110988-d3e0ec80-565a-11ea-9559-68b2728c8e59.PNG)

* Files/Directories included:
  1. **data**: The directory contains the main data for our project.
  2. **network1**: The directory includes Python files for 2 layer neural network.
  3. **network2**: The directory includes Python files for L layer neural network.(we are using 4-Layer Neural Network)

## Deep Learning For Images:
  
* Computers are able to perform computations on numbers and is unable to interpret images in the way that we do. We have to somehow convert the images to numbers for the computer to understand.

* There are two common ways to do this in Image Processing:

  1. **Using Greyscale**:
      * The image will be converted to greyscale (range of gray shades from white to black) the computer will assign each pixel a value based on how dark it is. All the numbers are put into an array and the computer does computations on that array.
      * This is how the number 8 is seen on using Greyscale:
          ![1_-xaK2HVoN-zI4rNXKCtjew](https://user-images.githubusercontent.com/33928040/75111039-608baa80-565b-11ea-8743-4f95246a18da.gif)
      * We then feed the resulting array into the computer:
          ![8](https://user-images.githubusercontent.com/33928040/75111052-97fa5700-565b-11ea-8fc7-908288291759.PNG)
  
  2. **Using RGB Values**:
      * Colors could be represented as RGB values (a combination of red, green and blue ranging from 0 to 255). Computers could then extract the RGB value of each pixel and put the result in an array for interpretation.
      * When the computer interprets a new image, it will convert the image to an array by using the same technique, which then compares the patterns of numbers against the already-known objects.
      * The computer then allots confidence scores for each class. The class with the highest confidence score is usually the predicted one.
      ![rgb](https://user-images.githubusercontent.com/33928040/75111086-ffb0a200-565b-11ea-8523-9ce34af0d48d.PNG)

**Reference**: https://www.coursera.org/learn/neural-networks-deep-learning/home/welcome
