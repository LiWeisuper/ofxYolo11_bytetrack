#pragma once
#include "ofMain.h"
#include <iostream>
#include <string>
#include "infer.h"
#include "BYTETracker.h"



class ofxYolo11_bytetrack
{
public:
	ofxYolo11_bytetrack(const std::string& enginePath = "onnx_model/yolo11s.plan");
	~ofxYolo11_bytetrack();

	bool setup();
	cv::Mat update(cv::Mat frame);
	void draw();


public:


private:
	YoloDetector detector;
	BYTETracker tracker;
};
