#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include "popt_pp.h"

using namespace std;
using namespace cv;

int main(int argc, char const *argv[])
{
  char* johnny_stereo_file;
  char* leftimg_filename;
  char* rightimg_filename;
  char* out_file;
  float focal_length;
  int field_of_view;
  char* leftout_filename;
  char* rightout_filename;

  static struct poptOption options[] = {
    { "johnny_stereo_file",'v',POPT_ARG_STRING,&johnny_stereo_file,0,"Johnny stereo calibration","STR" },
    { "leftimg_filename",'l',POPT_ARG_STRING,&leftimg_filename,0,"Left image path","STR" },
    { "rightimg_filename",'r',POPT_ARG_STRING,&rightimg_filename,0,"Right image path","STR" },
    { "out_file",'o',POPT_ARG_STRING,&out_file,0,"Output calibration filename (YML)","STR" },
    { "leftout_filename",'L',POPT_ARG_STRING,&leftout_filename,0,"Left undistorted image path","STR" },
    { "rightout_filename",'R',POPT_ARG_STRING,&rightout_filename,0,"Right undistorted image path","STR" },

    POPT_AUTOHELP
    { NULL, 0, 0, NULL, 0, NULL, NULL }
  };

  POpt popt(NULL, argc, argv, options, 0);
  int c;
  while((c = popt.getNextOpt()) >= 0) {}

  printf("Starting Calibration\n");
  FileStorage fsj(johnny_stereo_file, FileStorage::READ);

  Mat K1, K2, R, T;
  Mat D1, D2;
  Mat img1 = imread(leftimg_filename, IMREAD_COLOR);
  Mat img2 = imread(rightimg_filename, IMREAD_COLOR);

  fsj["R"] >> R;
  fsj["T"] >> T;

  cout << "Rotation matrix: " << endl << R << endl
     << "Translation vector: " << endl << T << endl;

  FileNode features = fsj["Cam_Params"];
  FileNodeIterator it = features.begin(), it_end = features.end();
  int idx = 0;

  // iterate through a sequence using FileNodeIterator
  for( ; it != it_end; ++it, idx++ )
  {
      cout << "Cam_Params #" << idx << ": " << endl;
      cout << "resolution_x = " << (int)(*it)["resolution_x"] << ", resolution_y = " << (int)(*it)["resolution_y"] << endl;
      cout << "focal length = " << (float)(*it)["f"] << ", Field of View = " << (int)(*it)["fov"] << endl;
      focal_length = roundf((float)(*it)["f"] * 100) / 100;
      field_of_view = (int)(*it)["fov"];
  }

  K1 = (Mat_<float>(3,3) << focal_length, 0, 0, 0, focal_length, 0, 0, 0, 1);
  K2 = (Mat_<float>(3,3) << focal_length, 0, 0, 0, focal_length, 0, 0, 0, 1);
  D1 = (Mat_<float>(1,5) << 0, 0, 0, 0, 0);
  D2 = (Mat_<float>(1,5) << 0, 0, 0, 0, 0);

  cout << "Camera matrix K1: " << endl << K1 << endl;
  cout << "Camera matrix K2: " << endl << K2 << endl;
  cout << "Camera matrix D1: " << endl << D1 << endl;
  cout << "Camera matrix D2: " << endl << D2 << endl;

  printf("Starting Rectification\n");

  Mat R1, R2, P1, P2, Q;
  stereoRectify(K1, D1, K2, D2, img1.size(), R, T, R1, R2, P1, P2, Q);

  FileStorage fs1(out_file, cv::FileStorage::WRITE);
  fs1 << "K1" << K1;
  fs1 << "K2" << K2;
  fs1 << "D1" << D1;
  fs1 << "D2" << D2;
  fs1 << "R" << R;
  fs1 << "T" << T;

  fs1 << "R1" << R1;
  fs1 << "R2" << R2;
  fs1 << "P1" << P1;
  fs1 << "P2" << P2;
  fs1 << "Q" << Q;

  printf("Done Rectification\n");

  Mat lmapx, lmapy, rmapx, rmapy;
  Mat imgU1, imgU2;

  initUndistortRectifyMap(K1, D1, R1, P1, img1.size(), CV_32F, lmapx, lmapy);
  initUndistortRectifyMap(K2, D2, R2, P2, img2.size(), CV_32F, rmapx, rmapy);
  remap(img1, imgU1, lmapx, lmapy, INTER_LINEAR);
  remap(img2, imgU2, rmapx, rmapy, INTER_LINEAR);

  imwrite(leftout_filename, imgU1);
  imwrite(rightout_filename, imgU2);

  fsj.release();
  fs1.release();
}
