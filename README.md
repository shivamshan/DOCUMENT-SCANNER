# DOCUMENT-SCANNER
## This is a project that aims to mimic the modern day mobile scanning applications.
The code is written in python and makes use of OpenCV to create an application that can be used to scan documents like class notes,cheques,memos,
reminder notes,etc.</br>
</br>
This project takes advantage of Opn CV's image processing and transformation capabilities to make a document scanning application.</br>
This is done by first extracting the **contours** from the image and then use them to resixe and align the image using a technique called
**FOUR-POINT TRANSFORM.**</br>
</br>
## 1. Orignal image
<img src="https://github.com/shivamshan/DOCUMENT-SCANNER/blob/master/1.png">
</br>

## 2. Image after preprocessing
<img src="https://github.com/shivamshan/DOCUMENT-SCANNER/blob/master/2.png">
</br>

## 3. Image along with contour
<img src="https://github.com/shivamshan/DOCUMENT-SCANNER/blob/master/3.png">
</br>

## 4. Image after FOUR-POINT TRANFORM
### Basically the image is resized and aligned to show only the relevant information.
<img src="https://github.com/shivamshan/DOCUMENT-SCANNER/blob/master/4.png">

(Notice how the image sort of seems **cropped** to show only the note and not the background.)
</br>

## 5. Final scanned image to be displayed
<img src="https://github.com/shivamshan/DOCUMENT-SCANNER/blob/master/5.png">
</br>
</br>
**For more details and examples, view the .ipynb file.**
</br>
</br>
Also the code for FOUR-POINT TRANSFORM TECHNIQUE and the inspiration for this project was acquired from<a href="https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/">pyimagesearch.</a>
