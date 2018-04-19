#include <iostream>
#include <cmath>
#include <ctype.h>

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

const int MAX_COUNT = 500;
int main(int argc, char ** argv)
{
    TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS,20,0.001);
    Size subPixWinSize(10,10);
    Size winSize(40,40);
    if(argc != 2)
    {
        std :: cout << "Unknown input " << std :: endl;
        return 0;
    }
    VideoCapture cap;
    cap.open(argv[1]);
    if(!cap.isOpened())
    {
        std :: cout << "Cannot open video flow" << std :: endl;
        return 0;
    }
    namedWindow("OpticalFlow");
    Mat gray,prevGray,image,frame;
    vector<Point2f> points[2];
    vector<Point2f> pointsTmp;

    while(1)
    {
        cap >> frame;
        if(frame.empty())
            break;
        image = frame.clone();
        cvtColor(image,gray,COLOR_RGB2GRAY);
        if(points[0].size() < 200)
        {
            goodFeaturesToTrack(gray,pointsTmp,MAX_COUNT,0.01,10,Mat(),3,3,0,0.4);
            cornerSubPix(gray,pointsTmp,subPixWinSize,Size(-1,-1),termcrit);
            for(int j = 0; j < pointsTmp.size(); j++)
            {
                bool in(false);
                for(int i = 0; i < points[0].size(); i++)
                {
                    if(pointsTmp[j] == points[0][i])
                    {
                        in = true;
                        break;
                    }
                }
                if(!in)
                    points[0].push_back(pointsTmp[j]);
            }
        }
        vector<uchar> status;
        vector<float> err;
        if(prevGray.empty())
            prevGray = gray.clone();
        calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
                                         5, termcrit, 0, 0.001);
        int k = 0;
        for(int i = k = 0; i < points[1].size(); i++)
        {
            if(!status[i])
                continue;
            points[1][k++] = points[1][i];
            line(image,points[1][i],points[1][i] + 10*(points[1][i] - points[0][i])/norm((points[1][i] - points[0][i])),Scalar(100,100,255),2);
        }
        points[1].resize(k);
        imshow("OpticalFlow",image);
        waitKey(10);
        std :: swap(points[1],points[0]);
        cv :: swap(prevGray,gray);
    }
    return 0;
}
