# Project: Search and Sample Return
## Writeup / README
### This project is based on the NASA Sample Return Challenge to locate the samples of interest, pick them up and return them to a starting point.

---
---

#### Notebook Analysis  

###### 1. At first, I run the Notebook as it was after making the necessary modification for downloading imageio plugins. The notebook run a sample test file which was already present in the test_dataset folder.  

###### 2. Then, I filled the `process_image()` function with the function calls of the perspective transform, color thresholding, conversion of image pixels to rover centric coordinates and to world coordinates. World map is created by superimposing the calibration image of map with the explored region and found samples of interest.
###### As the number of images in the test_dataset are finite, data.count is included which is incremented at the end of every function call. 

###### 3. After the module was working correctly for navigable terrain, I inserted the funtions `obstacles()` and `rocks()`. 
###### The function `obstacles()` is derived from the function `color_thresh()` and identifies the pixel values RGB < 160 to determine the obstacles in the path of the rover. 
###### The range of color yellow is defined as an array in `rocks()` function to determine the samples of interest (pictured as yellow rocks). RGB pixel values are converted to HSV values in order to be compared with the range of color yellow. 
###### The two functions `obstacles()` and `rocks()` were first individually tested on the calibration images before updating the `process_image()` function. 
![alt text](https://drive.google.com/file/d/0Bz8idi001SUZLUJOazlxRXgxeTg/view?usp=sharing "rock_image")
![alt text](https://drive.google.com/file/d/0Bz8idi001SUZVlFaVmljRk5vVjQ/view?usp=sharing "rock_transformed_image")
###### The `process_image()` function was updated later with the function calls of `obstacles()` and `rocks()` and converting the image pixels into world coordinates.

###### 4. Finally, when all the functions were working correctly, I captured a small test_dataset and made an output video using moviePy. The output video is present in the same .zip package.

---

#### Automatic and Navigable Mapping

###### 1. `perception.py` - `perception.py` takes data input from the class `RoverState()` (coordinates, yaw, pitch, roll, throttle, mode, images etc) and processes the images and coordinates so that `decision.py` may take necessary action on the Rover.
###### In this program, I filled the function `perception_step()` with all the function calls present in the `process_image()` function of Notebook. I also updated the rover vision images for navigable terrain, obstacles and rocks (samples of interest) for display on the world map on the right side of screen. The rocks are displayed on the world map as small yellow colored dots. 

###### 2. `decision.py` - `decision.py` takes care of the Rover decisions based on `perception.py`. 
###### I modified the function `decision_step()` function in this program to make the Rover more intelligent. Now the Rover is able to apply brakes if there is a sample of interest nearby and pick them up. Also it counts the number of samples found. 


---

#### Possible improvements in the project

###### 1. The Rover travels in the center of the navigable terrain. So, if the samples are too near to the obstacles, then the Rover avoids them completely. Changes may be done in `decision.py` to solve this.

###### 2. Sometimes, the Rover traverses the explored path again and again. For solving this issue, changes needs to be done in `decision.py`.

---
---






