/*
 * tuFace
 *
 * Find and match faces in all of the given input IP webcam streams.
 *
 *
 * 
 * Copyright (c) 2014. Brian Rudy <brudy[at]praecogito[dot]com>.
 *
 * Original Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
 * Released to public domain under terms of the BSD Simplified license.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the organization nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 *   See <http://www.opensource.org/licenses/bsd-license>
 *
 * References
 * -creating multi-threaded camera capture with Boost
 * 	https://aaka.sh/patel/2013/06/28/live-video-webcam-recording-with-opencv
 *
 * -Original video face recognition sample from libfacerec
 * 	https://github.com/bytefish/libfacerec/blob/master/samples/facerec_video.cpp
 *
 */


#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/face.hpp"
#include <opencv2/tracking.hpp>
#include "opencv2/core/ocl.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

#include <getopt.h>

// include the Boost headers for threading
#include "boost/thread.hpp"
// include the Boost headers for high precision system time
#include "boost/date_time/posix_time/posix_time.hpp"

using namespace cv;
using namespace cv::face;
using namespace boost::posix_time;
using namespace std;

// max number of people to search for
#define MAX_PEOPLE	5
#define NO_ARG		0
#define REQUIRED_ARG	1 
#define OPTIONAL_ARG	2

// name of people
string people[MAX_PEOPLE];


static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
	cerr << "No valid input file was given, please check the given filename." << endl;
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

static void print_usage(const string& ourname) {
	cout << "usage: " << ourname << " --cascade </path/to/haar_cascade> --csv </path/to/csv.ext> --url <cam URL>" << endl;
	cout << "\t --cascade </path/to/haar_cascade> -- Path to the Haar Cascade for face detection." << endl;
	cout << "\t --csv </path/to/csv.ext> -- Path to the CSV file with the face database." << endl;
	cout << "\t --url <cam URL> -- The IP webcam URL to grab frames from." << endl;
	cout << "\t\t Note: Provide as many --url arguments as you have camera feeds" << endl;
	cout << "\t --help -- Print this help text." << endl;
	exit(1);
}

// Code for capture thread
void captureFunc(vector<VideoCapture> *capture) {
	size_t size = 10;
	vector<ptime> initialLoopTimestamp(size), nextFrameTimestamp(size), currentFrameTimestamp(size), finalLoopTimestamp(size);
	vector<time_duration> td(size), td1(size);
	int framerate = 15;

	//initialize initial timestamps
	for (unsigned cap_index=0; cap_index<(*capture).size(); cap_index++) {
		nextFrameTimestamp[cap_index] = microsec_clock::local_time();
		currentFrameTimestamp[cap_index] = nextFrameTimestamp[cap_index];
		td[cap_index] = (currentFrameTimestamp[cap_index] - nextFrameTimestamp[cap_index]);
	}

	for(;;){
		for (unsigned cap_index=0; cap_index<(*capture).size(); cap_index++) {
			// wait for X microseconds until 1second/framerate time has passed after previous frame grab
			while(td[cap_index].total_microseconds() < 1000000/framerate){
				//determine current elapsed time
				currentFrameTimestamp[cap_index] = microsec_clock::local_time();
				td[cap_index] = (currentFrameTimestamp[cap_index] - nextFrameTimestamp[cap_index]);
			}
			// grab from camera as fast as possible. We will retreive() it later.
			// If we don't do it this way, because the face detection stuff is so slow
			// the capture buffer will fill up after a few seconds, and we won't be able 
			// to read() it anymore.

			//determine time at start of write
			initialLoopTimestamp[cap_index] = microsec_clock::local_time();

			(*capture)[cap_index].grab();
			// add 1second/framerate time for next loop pause
			nextFrameTimestamp[cap_index] = nextFrameTimestamp[cap_index] + microsec(1000000/framerate);
			// reset time_duration so while loop engages
			td[cap_index] = (currentFrameTimestamp[cap_index] - nextFrameTimestamp[cap_index]);

			// determine and print out delay in ms, should be less than 1000/FPS
			// occasionally, if delay is larger than said value, correction will occur
			// if delay is consistently larger than said value, the network/system is not fast
			// enough to run at that frame rate.
			//finalLoopTimestamp[cap_index] = microsec_clock::local_time();
			//td1[cap_index] = (finalLoopTimestamp[cap_index] - initialLoopTimestamp[cap_index]);
			//int delayFound = td1[cap_index].total_milliseconds();
			//cout << "Delay from cam " << cap_index << " = " << delayFound << endl;
		}
	}
}


int main(int argc, char **argv) {
    string fn_haar, fn_csv;
    vector<string> cam_url;

    const char *s_option="hc:v:u:";
    const struct option l_option[]={
	{ "help",	NO_ARG,		NULL, 'h'},
	{ "cascade",	REQUIRED_ARG,	NULL, 'c'},
	{ "csv",	REQUIRED_ARG,	NULL, 'v'},
	{ "url",	REQUIRED_ARG,	NULL, 'u'}
    };

    unsigned n_opt;
    do {
	n_opt=getopt_long(argc,argv,s_option,l_option,NULL);
	switch(n_opt){
		case 'h':
		case '?':
			print_usage(argv[0]);
			break;
		case 'c':
			fn_haar = optarg;
			break;
		case 'v':
			fn_csv = optarg;
			break;
		case 'u':
			cam_url.push_back(optarg);
			break;
		case -1:
		default:
			break;
	}
    } while (n_opt!=-1);


    //ocl::setUseOpenCL(false);
    printf("OpenCL: %s\n", ocl::useOpenCL() ? "ON" : "OFF");
    people[0] 	= "Unknown";
    people[1] 	= "Unknown";
    people[2] 	= "Brian";
    people[3] 	= "Helen";
    people[4] 	= "Brionna";

    // These vectors hold the images and corresponding labels:
    vector<Mat> images;
    vector<int> labels;
    // Read in the data (fails if no valid input filename is given, but you'll get an error message):
    try {
	cout << "Reading CSV." << endl;
        read_csv(fn_csv, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }
    // Get the height from the first image. We'll need this
    // later in code to reshape the images to their original
    // size AND we need to reshape incoming faces to this size:
    int im_width = images[0].cols;
    int im_height = images[0].rows;
    // Create a FaceRecognizer and train it on the given images:
    Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
    //Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
    //Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();
    model->train(images, labels);
    cout << "Done training." << endl;
    // That's it for learning the Face Recognition model. You now
    // need to create the classifier for the task of Face Detection.
    // We are going to use the haar cascade you have specified in the
    // command line arguments:
    CascadeClassifier haar_cascade;
    haar_cascade.load(fn_haar);

    // initialize the tracker
    Ptr<Tracker> tracker = Tracker::create( "MEDIANFLOW" );
    if( tracker == NULL ) {
	cout << "***Error in the instantiation of the tracker...***\n";
	return -1;
    }
    bool trackerInitialized = false;
    bool trackerActive = false;
    Rect2d trackerBoundingBox;

    // Get a handle to the Video device:
    vector<VideoCapture> cap;
    // We need to declare the Mats outside of the camera loop for this to work with more than one camera
    size_t size = 10;
    vector<Mat> frame(size);
    vector<Mat> original(size);
    vector<Mat> gray(size);

    for (unsigned cam_index=0; cam_index<cam_url.size(); cam_index++) {
	// Check if we can use this device at all:
	VideoCapture icap;
	if(!icap.open(cam_url.at(cam_index))) {
		cerr << "Capture source " << cam_url.at(cam_index) << " cannot be opened." << endl;
		return -1;
	} else {
		cap.push_back(icap);
		cout << "Camera " << cam_index << ": (" << cam_url.at(cam_index) << ") is now open." << endl;
		std::stringstream cam_index_str;
                cam_index_str << cam_index;
                string cam_frame_title = "tuFace-" + cam_index_str.str();
                //namedWindow(cam_frame_title, WINDOW_OPENGL);
                namedWindow(cam_frame_title, WINDOW_AUTOSIZE);
	}
    }

    // start thread to begin grabbing
    boost::thread captureThread(captureFunc, &cap);

    // Give the cameras a chance to start streaming or we will get empty frames for a while
    usleep(10000);

    // All the cameras are now open, start looking for faces
    for(;;) {
    	for (unsigned cap_index=0; cap_index<cap.size(); cap_index++) {
		double t = 0;
		// decode the latest frame that we grabbed earlier
		if(!cap.at(cap_index).retrieve(frame[cap_index])) {
			cout << "No frame from cam " << cap_index << endl;
			if(!cap.at(cap_index).isOpened()) {
				cout << "Cam " << cap_index << " is no longer open." << endl;
			}
			cv::waitKey(1);
			continue;
		}
		t = (double)getTickCount();
        	// Clone the current frame:
        	original[cap_index] = frame[cap_index].clone();
        	// Convert the current frame to grayscale:
        	cvtColor(original[cap_index], gray[cap_index], COLOR_BGR2GRAY);
		// We need to equalize the histo to reduce the impact of lighting
		equalizeHist(gray[cap_index],gray[cap_index]);
        	// Find the faces in the frame:
        	vector< Rect_<int> > faces;
        	haar_cascade.detectMultiScale(gray[cap_index], faces);
        	//haar_cascade.detectMultiScale(gray, faces, 1.1, 3, CASCADE_SCALE_IMAGE, Size(80,80));
        	// At this point you have the position of the faces in
        	// faces. Now we'll get the faces, make a prediction and
        	// annotate it in the video. Cool or what?
        	for(int i = 0; i < faces.size(); i++) {
            		// Process face by face:
            		Rect face_i = faces[i];
            		// Crop the face from the image. So simple with OpenCV C++:
            		Mat face = gray[cap_index](face_i);
            		// Resizing the face is necessary for Eigenfaces and Fisherfaces. 
            		Mat face_resized;
            		cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
            		// Now perform the prediction, see how easy that is:
            		int prediction = -1;
            		double predicted_confidence = 0.0;
            		model->predict(face_resized,prediction,predicted_confidence);
            		// And finally write all we've found out to the original image!
            		if (predicted_confidence > 250.0) { //Fischerfaces
            		//if (predicted_confidence > 4500.0) { //Eigenfaces
            			// If the tracker hasn't been initialized, do so on this face
				if(!trackerInitialized) {
					trackerBoundingBox = face_i;
            				if(!tracker->init(original[cap_index], trackerBoundingBox)) {
						cout << "***Could not initialize tracker...***\n";
						return -1;
					}
					trackerInitialized = true;
				}

            			rectangle(original[cap_index], face_i, Scalar(0,255,0), 1);
            			// Create the text we will annotate the box with:
            			std::stringstream confidence;
            			confidence << predicted_confidence;
	    			string box_text = "Id="+people[prediction] +", conf="+confidence.str();
            			// Calculate the position for annotated text (make sure we don't
            			// put illegal values in there):
            			int pos_x = std::max(face_i.tl().x - 10, 0);
            			int pos_y = std::max(face_i.tl().y - 10, 0);
            			// And now put it into the image:
            			if (i == 0) {
					t = (double)getTickCount() - t;
					double fps = getTickFrequency()/t;
					static double avgfps = 0;
					static int nframes = 0;
					nframes++;
					double alpha = nframes > 50 ? 0.01 : 1./nframes;
					avgfps = avgfps*(1-alpha) + fps*alpha;
					putText(original[cap_index], format("OpenCL: %s, fps: %.1f", ocl::useOpenCL() ? "ON" : "OFF", avgfps), Point(50, 30),
						FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,255,0), 2);
	    			}
            			putText(original[cap_index], box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, Scalar(0,255,0), 2.0);
	    		} else {
				rectangle(original[cap_index], face_i, Scalar(0,0,255), 1);
	    		}
			face.release();
			face_resized.release();
        	}
		std::stringstream cap_index_str;
		cap_index_str << cap_index;
		string cam_frame_title = "tuFace-" + cap_index_str.str();
		if(trackerInitialized) {
			// Update the tracker
			if(tracker->update(original[cap_index], trackerBoundingBox)) {
				// Draw the tracker bounding box
				rectangle(original[cap_index], trackerBoundingBox, Scalar(255,0,0), 2, 1);
			} else {
				cout << "We lost the face we were tracking. Resetting to track a new face." << endl;
				trackerInitialized = false;
				// because we cannot simply init() again to reset, we have to recreate the tracker instance
				tracker = Tracker::create( "MEDIANFLOW" );
			}
		}
        	// Show the result:
        	imshow(cam_frame_title, original[cap_index]);
		frame[cap_index].release();
		original[cap_index].release();
		gray[cap_index].release();

		if(cv::waitKey(1) >= 0) break;
	}
    }
    return 0;
}
