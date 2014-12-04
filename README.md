tuFace
======

An OpenCL-enabled face recognition app for multiple IP webcams built on OpenCV.

<img src="http://www.praecogito.com/photobucket/tuFace-screenshot-11-23-2014.png" alt="tuFace Screenshot">

##Description
  This project is intended to provide face recognition from multiple IP webcam streams to serve as part of a larger system that provides occupant detection and identification. It is OpenCL-enabled to take advantage of the boost in performance over traditional CPUs enabled by GPUs/APUs. Since OpenCV doesn't yet support aggregation of OpenCL resources (e.g. sharing the resources of multiple GPUs from a single application), it is possible to run multiple versions of this application, each using a single OpenCL device by means of the OPENCV_OPENCL_DEVICE environment variable. See [here](http://docs.opencv.org/modules/ocl/doc/introduction.html) for more details.

This code is alpha-level at best, and users should have a working knowledge of the component technologies involved (C++, OpenCV, Haar cascades, Eigenfaces/Fisherfaces/LBP, etc) before attempting any modifications yourself.

##Limitations
The current version of tuFace spawns a seperate video capture thread to grab frames from the video streams as fast as possible.  As a result, if any of the camera feeds takes extra time to grab(), it will slow down the capture thread and the frame rate of the other cameras will also drop. In a future version each camera feed will get it's own capture thread.
  
If we don't multi-thread and instead attempt to serialize grabbing->decoding->processing in a single thread, the FFMPEG video buffer that OpenCV uses for the IP camera streams will fill up because we are not pulling the frames out quickly enough. OpenCV 3.0 beta doesn't appear to have the ability to manipulate the buffer behavior, adjust the frame rate, or seek forward in the stream for MJPEG streams, so the only way to recover is by re-opening the MJPEG stream over and over. If you want to see what this looks like, have a look at the [IP-cam-single-thread branch of tuFace](https://github.com/rudybrian/tuFace/tree/IP-cam-single-thread). This may not be obvious for a single camera feed, but once multiple camera feeds are being processed, this issue will appear fairly quickly (in my case this took a ~20 seconds from a fast camera).

##Requirements
* [OpenCV 3.0 Beta](https://github.com/Itseez/opencv/tree/3.0.0-beta)
* The face module from [OpenCV 3.0 Beta Contrib](https://github.com/Itseez/opencv_contrib/tree/3.0.0-beta)
* Boost

##Build Instructions
* cd into source directory
* *cmake .*
* *make*

##Face Database Preparation
**Recognition accuracy is entirely dependant on the quality of the face database.** If you are not familiar with this process, I suggest reading up on this from the reference links below.

##Execution
Run the app with somethng like:
```
OPENCV_OPENCL_DEVICE=':dGPU:2' ./tuFace --cascade /usr/local/share/OpenCV/lbpcascades/lbpcascade_frontalface.xml --csv csv7.txt --url "http://user:password@192.168.1.1/cgi/mjpg/mjpg.cgi?bogus=foo.mjpg" --url "http://user:password@192.168.1.2/cgi/mjpg/mjpg.cgi?bogus=foo.mjpg"
```

##References
* http://thinkrpi.wordpress.com/2013/04/03/step-5-prepare-photos/
* http://www.shervinemami.info/faceRecognition.html
* http://thinkrpi.wordpress.com/2013/06/15/opencvpi-cam-step-7-face-recognition/
* http://docs.opencv.org/trunk/modules/contrib/doc/facerec/
