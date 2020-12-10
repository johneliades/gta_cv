# lane_detection

A simple lane detection algorithm for GTA V written in python using openCV that 
decently detects lanes in most weather conditions using Canny, GaussianBlur and 
HoughLines with dynamic threshold adjustment depending on weather. The amazing
part being that it can run without almost any modifications for any other game 
with lanes. Its uses? No idea, maybe self driving neural network with reinforcement 
learning? Maybe not a single one. Just found it a fun project to do xD.

<p align="center">
  <img src="https://github.com/johneliades/gta_cv/blob/main/lane_detection/preview.gif" alt="animated" />
</p>

## Clone

Clone the repository locally by entering the following command:
```
git clone https://github.com/johneliades/gta_cv.git
```
Or by clicking on the green "Clone or download" button on top and then decompressing the zip.

## Run

Switch GTA V to windowed mode and then open a terminal in the cloned folder and enter:

```
cd lane_detection
python3 main.py
```

## Author

**Eliades John** - *Developer* - [Github](https://github.com/johneliades)
