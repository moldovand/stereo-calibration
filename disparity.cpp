#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <string.h>
#include "popt_pp.h"

using namespace cv;
using namespace std;

int main(int argc, char const *argv[])
{
  char* leftimg_filename;
  char* rightimg_filename;
  char* method;
  char* disparity_filename;

  static struct poptOption options[] = {
    { "leftimg_filename",'l',POPT_ARG_STRING,&leftimg_filename,0,"Left imgage path","STR" },
    { "rightimg_filename",'r',POPT_ARG_STRING,&rightimg_filename,0,"Right image path","STR" },
    { "method",'m',POPT_ARG_STRING,&method,0,"Disparity calculation method","STR" },
    { "disparity_filename",'D',POPT_ARG_STRING,&disparity_filename,0,"Left undistorted imgage path","STR" },
    POPT_AUTOHELP
    { NULL, 0, 0, NULL, 0, NULL, NULL }
  };

  POpt popt(NULL, argc, argv, options, 0);
  int c;
  while((c = popt.getNextOpt()) >= 0) {}

    Mat g1, g2;
    Mat img1 = imread(leftimg_filename, IMREAD_COLOR);
    Mat img2 = imread(rightimg_filename, IMREAD_COLOR);

    Mat disp, disp8;

    int ndisparities = 50;   /**< Range of disparity */
    int SADWindowSize = 1; /**< Size of the block window. Must be odd */
    Ptr<StereoBM> sbm = StereoBM::create( ndisparities, SADWindowSize );
    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,    //int minDisparity
                                    32,     //int numDisparities
                                    5,      //int SADWindowSize
                                    200,    //int P1 = 0
                                    500,   //int P2 = 0
                                    10,     //int disp12MaxDiff = 0
                                    1,     //int preFilterCap = 0
                                    0,      //int uniquenessRatio = 0
                                    60,    //int speckleWindowSize = 0
                                    7,     //int speckleRange = 0
                                    true);  //bool fullDP = false

    cvtColor(img1, g1, COLOR_BGR2GRAY);
    cvtColor(img2, g2, COLOR_BGR2GRAY);

    if (!(strcmp(method, "BM")))
    {
        sbm->compute(g1, g2, disp);
    }
    else if (!(strcmp(method, "SGBM")))
    {
        sgbm->compute(g1, g2, disp);
    }

    normalize(disp, disp8, 0, 255, NORM_MINMAX, CV_8U);

    imshow("left", img1);
    imshow("right", img2);
    imshow("disp", disp8);

    imwrite(disparity_filename, disp8);

    waitKey(0);

    return(0);
}
