#include "ofxYolo11_bytetrack.h"

// 需要跟踪的类别，可以根据自己需求调整，筛选自己想要跟踪的对象的种类（以下对应COCO数据集类别索引）
std::vector<int>  trackClasses{ 0};  // 0:person, bicycle, car, motorcycle, bus, truck 73:book
bool isTrackingClass(int class_id) {
	for (auto& c : trackClasses) {
		if (class_id == c) return true;
	}
	return false;
}

ofxYolo11_bytetrack::ofxYolo11_bytetrack(const std::string& enginePath)
	: detector(enginePath, 0, 0.45, 0.01), tracker(60, 30)
{
}

ofxYolo11_bytetrack::~ofxYolo11_bytetrack()
{
}

bool ofxYolo11_bytetrack::setup()
{

	return true;
}

cv::Mat ofxYolo11_bytetrack::update(cv::Mat frame)
{
	cv::Mat img;

	if (!frame.empty()) {

		frame.copyTo(img);

		// yolo inference
		std::vector<Detection> res = detector.inference(img);

		// yolo output format to bytetrack input format, and filter bbox by class id
		std::vector<Object> objects;
		for (size_t j = 0; j < res.size(); j++) {
			float* bbox = res[j].bbox;
			float conf = res[j].conf;
			int classId = res[j].classId;

			if (isTrackingClass(classId)) {
				cv::Rect_<float> rect(bbox[0], bbox[1], (bbox[2] - bbox[0]), (bbox[3] - bbox[1]));
				Object obj{ rect, classId, conf };
				objects.push_back(obj);
			}
		}

		// track
		std::vector<STrack> output_stracks = tracker.update(objects);

		for (int i = 0; i < output_stracks.size(); i++)
		{
			std::vector<float> tlwh = output_stracks[i].tlwh;

			if (tlwh[2] * tlwh[3] > 20)
			{
				cv::Scalar s = tracker.get_color(output_stracks[i].track_id);
				cv::putText(img, cv::format("%d", output_stracks[i].track_id), cv::Point(tlwh[0], tlwh[1] - 5),
					0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
				cv::rectangle(img, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
			}
		}
	}

	return img;

}


void ofxYolo11_bytetrack::draw()
{


}
