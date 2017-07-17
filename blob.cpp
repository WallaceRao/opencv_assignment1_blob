#include <stdio.h>
#include <chrono>
#include <ctime>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <set>
#include <map>
#include <stdlib.h>

#define Mpixel(image,x,y) (( uchar *)(((image).data)+(y)*((image).step)))[(x)]


/*********************************************************************************************
 * compile with:
 * g++ -O3 -o main blob.cpp -std=c++11 `pkg-config --cflags --libs opencv`
*********************************************************************************************/



using namespace std ;
using namespace cv ;
using namespace chrono;

Mat frame; // Image from camera


// Get median value from an array
int getMedian(int a[], int size)
{
    for(int pass=0;pass<=size/2;pass++)
     {
        for(int k=0;k<size-pass-1;k++)
           if(a[k]>a[k+1])
           {
                int temp=a[k];
                a[k]=a[k+1];
                a[k+1]=temp;
           }
    }
    return a[4];
}

/*
 * Customized median filter, window size is 3 * 3.
 */

bool myMedianBlur(Mat inImg, Mat &outImg)
{
    outImg.create(inImg.size(), CV_8UC1);
    outImg.setTo(Scalar(0));

    Mat imageGrey = inImg;
    if(inImg.type() != CV_8UC1)
        cvtColor(inImg, imageGrey, CV_BGR2GRAY);
    for (int x = 0; x < imageGrey.cols; x++)
        for (int y = 0; y < imageGrey.rows;y ++)
            if(x == 0 || y == 0 || x == imageGrey.cols -1 || y == imageGrey.rows -1)
            {
              continue;
            }
            else
            {
                int a[9];
                a[0] = Mpixel(imageGrey, x-1, y-1);//  (imageGrey.at<uchar>(y-1, x-1));
                a[1] = Mpixel(imageGrey, x, y-1);//(imageGrey.at<uchar>(y, x-1));
                a[2] = Mpixel(imageGrey, x+1, y-1);//(imageGrey.at<uchar>(y+1, x-1));
                a[3] = Mpixel(imageGrey, x-1, y);//(imageGrey.at<uchar>(y-1, x));
                a[4] = Mpixel(imageGrey, x, y);//(imageGrey.at<uchar>(y, x));
                a[5] = Mpixel(imageGrey, x+1, y);//(imageGrey.at<uchar>(y+1, x));
                a[6] = Mpixel(imageGrey, x-1, y+1);//(imageGrey.at<uchar>(y-1, x+1));
                a[7] = Mpixel(imageGrey, x, y+1);//(imageGrey.at<uchar>(y, x+1));
                a[8] = Mpixel(imageGrey, x+1, y+1);//(imageGrey.at<uchar>(y+1, x+1));
                Mpixel(outImg,x,y) = getMedian(a, 9);//*(buffer+4);

            }
    return true;
}


/*
 * Count the blobs, highlight every blob use random color if highlightBlob is true.
 */
int blobCount(Mat inImg,  Mat &outImage, bool highlightBlob)
{
    int counter = -1, s1, s2;
    int height = inImg.size().height;
    int width = inImg.size().width;

    /*
     * "A" is used to store pixel-> group(or blob) information. for example,
     " A[1,1] = a, means the pixel[1,1] belongs to group "a".
     */
    vector< vector<int> > A(height, vector<int>(width, -1));
    /*
     * mapGroup is used to store the group map relationships, for example,
     * mapGroup[a,b] means the group "a" has been redirected to group "b", and
     * every point in group "a" belongs to "b" now.
     * When combine two groups s1 and s2, we can simply specify mapGroup[s1] = s2,
     * thus the real compine operation could be avoided and the performance is improved.
     */
    map<int,int> mapGroup;
    for(int x = 1; x <width; x ++)
        for(int y = 1; y < height; y++)
            if(inImg.at<uchar>(y,x) != 0)
            {
                if(A[y-1][x] != -1 || A[y][x-1] != -1)
                {
                    s1 = A[y-1][x];
                    /*
                     * If group s1 has been redirected to another group, find out
                     * the final target it points to.
                     */
                    while(mapGroup.find(s1) != mapGroup.end() && s1 != mapGroup[s1])
                        s1 = mapGroup[s1];

                    s2 = A[y][x-1];
                    /*
                     * If group s2 has been redirected to another group, find out
                     * the final target it points to.
                     */
                    while(mapGroup.find(s2) != mapGroup.end() && s2 != mapGroup[s2])
                        s2= mapGroup[s2];
                    if(s1 != -1)
                    {
                        // Add pixel(y,x) to group s1.
                        A[y][x] = s1;
                    }
                    else if(s2 != -1)
                    {
                        // Add pixel(y,x) to group s2.
                        A[y][x] = s2;
                    }
                    if(s1 != -1 && s2 != -1 && s1 != s2)
                    {
                        // compine s1 and s2, make s2 point to s1.
                        mapGroup[s2] = s1;
                    }
                }
                else
                {
                    // Create a new group for this pixel, this new group is redirected to itself.
                    counter ++;
                    A[y][x] = counter;
                    mapGroup[counter] = counter;
                }
            }
    int blobCount = 0;
    set<int> groups;  // Store the groups that has never been redirected to another group.
    for(map<int,int>::iterator it = mapGroup.begin(); it != mapGroup.end(); ++it)
     {
        // If the group points to itself, that means the group still exist, otherwise it has been compined by "it->second".
        if(it->second == it->first)
            groups.insert(it->first);
    }
    blobCount = groups.size();

    char str[25];
    snprintf(str, sizeof(str), "Blob Count: %d", blobCount);
    putText(inImg,str,cvPoint(10, 30),CV_FONT_HERSHEY_DUPLEX,1.0f,CV_RGB(128,128,128));
    // If highlightBlob is true, hightlight every blob.
    if(highlightBlob)
    {
        outImage.create(inImg.size(),CV_8UC3);
        //  Traverse mapGroup and make every group points to its final target(the actual group).
        for(map<int,int>::iterator it = mapGroup.begin(); it != mapGroup.end(); ++it)
        {
            int group = it->first;
            int newGroup = it->second;
            while (mapGroup.find(newGroup) != mapGroup.end() && newGroup != mapGroup[newGroup])
            {
                newGroup =  mapGroup[newGroup];
            }
            // newGroup is the actual group, it should point to itself.
            if(newGroup != mapGroup[newGroup])
                cout << "error" << endl;
            mapGroup[group] = newGroup;
        }
        // groupToColor stores random color for every group.
        map<int, CvScalar> groupToColor;
        for(set<int>::iterator it = groups.begin(); it != groups.end(); ++it)
        {
            int group =  *it;
            RNG rng(group);
            int b = rng.uniform(0,255);
            int g = rng.uniform(0,255);
            int r = rng.uniform(0,255);
            groupToColor[group] = cvScalar(b , g, r);

        }
        for(int x = 0; x <width; x ++)
            for(int y = 0; y < height; y++)
            {
                // Find out the group that pixel belongs to.
                int group = A[y][x];
                if(-1 == group)
                {
                    // make it black if it do not belong to any group(blob).
                    outImage.at<Vec3b>(y,x)[0]=0;
                    outImage.at<Vec3b>(y,x)[1]=0;
                    outImage.at<Vec3b>(y,x)[2]=0;
                    continue;
                }
                int trueGroup = mapGroup[group];
                // Use the group color.
                CvScalar color = groupToColor[trueGroup];
                outImage.at<Vec3b>(y,x)[0]=color.val[0];
                outImage.at<Vec3b>(y,x)[1]=color.val[1];
                outImage.at<Vec3b>(y,x)[2]=color.val[2];

            }

        putText(outImage,str,cvPoint(10, 30),CV_FONT_HERSHEY_DUPLEX,1.0f,CV_RGB(128,128,128));
    }
    return blobCount;
}

/*
 * showCam captures frame from camera,  after median filer and
 * threshold filter, calculate the blob count and output the result.
 */

void showCam()
{
    VideoCapture cap;
    cap.open(0);
    if (!cap.isOpened())
        {
            cout << "Failed to open camera" << endl;
            return;
        }
    cout << "Opened camera" << endl;
    namedWindow("WebCam", 1);
    namedWindow("Highlight blob", 1);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

    cap >> frame;
    printf("frame size %d %d \n",frame.rows, frame.cols);
    int key=0;
    double fps=0.0;
    while (true)
    {
        system_clock::time_point start = system_clock::now();
        cap >> frame;
        if( frame.empty() )
        break;

        Mat oriImage = frame, borderImage, medianImage, thresImage, blobImage, destImage;
        oriImage = frame;

        if(frame.type() != CV_8UC1)
            cvtColor(frame, oriImage, CV_BGR2GRAY);

        borderImage.create(cv::Size(oriImage.size().height+2, oriImage.size().width+2), CV_8UC1);
        copyMakeBorder(oriImage,borderImage,1,1,1,1,BORDER_CONSTANT);

        // Add median filter below if the image has noise.
        myMedianBlur(borderImage, medianImage);  // median flter by 3 * 3 windows.
        threshold(medianImage, blobImage, 0, 255, CV_THRESH_BINARY|CV_THRESH_OTSU);

        int number = 0;
        /*
         * Calculate the blob count, the last parameter decides whether every blob should be highlighted.
         */
        number = blobCount(blobImage, destImage, true);
        imshow("Highlight blob", destImage);
        /*
         * Use blow line instead of the two lines above to improve the performance by avoiding highlighting.
         */
        //number = blobCount(blobImage, blobImage, false);

        char str[25];
        snprintf(str, sizeof(str), "blobs:%d", number);
        putText(frame,str,cvPoint(10, 30),CV_FONT_HERSHEY_DUPLEX,1.0f,CV_RGB(128,128,128));

        system_clock::time_point end = system_clock::now();
        double seconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        fps = 1000000/seconds;
        cout << "frames " << fps << " seconds " << seconds << endl;

        char printit[100];
        sprintf(printit,"frames: %2.1f",fps);
        putText(frame, printit, cvPoint(10,80), FONT_HERSHEY_PLAIN, 2, cvScalar(128,128,128), 2, 8);
        imshow("WebCam", frame);

        key=waitKey(1);
        if(key==113 || key==27)
            return;//either esc or 'q'

    }
}



int main(int argc, char** argv )
{
    // if there is no argument , exit
    if ( argc != 2)
    {
        cout << " needs 1 argument , e.g . image . jpg " << endl ;
        showCam();
        return 0;
    }
    /*
     * If user specify a static image, median filter the image, thresthod filter
     * the image and median filter again, then Gaussian filter, at last count the blobs and output the result.
     */

    namedWindow( "Original Image" , 0 ) ;

    Mat bufferImage[7];

    bufferImage[0]=imread(argv[1]);
    imshow( "Original Image" , bufferImage[0]);

    bufferImage[1] = bufferImage[0];
    if(bufferImage[0].type() != CV_8UC1)
        cvtColor(bufferImage[0], bufferImage[1], CV_BGR2GRAY);

    // make border
    bufferImage[2].create(cv::Size(bufferImage[1].size().height+2, bufferImage[1].size().width+2), CV_8UC1);
    copyMakeBorder(bufferImage[1],bufferImage[2],1,1,1,1,BORDER_CONSTANT);

    myMedianBlur(bufferImage[2], bufferImage[3]);
    namedWindow( "step1: Median Filter" , 0 ) ;
    imshow("step1: Median Filter", bufferImage[3]);

    threshold(bufferImage[3], bufferImage[4], 0, 255, CV_THRESH_BINARY|CV_THRESH_OTSU);
    namedWindow( "step2: threshold" , 0 );
    imshow("step2: threshold", bufferImage[4]);


    myMedianBlur(bufferImage[4], bufferImage[5]);
    namedWindow( "step3: Median Filter" , 0 );
    imshow("step3: Median Filter", bufferImage[5]);

    GaussianBlur(bufferImage[5], bufferImage[6], Size(7, 7), 0, 0 );//applying Gaussian filter
    namedWindow( "step4: GaussianBlur Filter" , 0 );
    imshow("step4: GaussianBlur Filter", bufferImage[6]);

    // count blobs and highlight blobs.
    Mat blobImage = bufferImage[6];
    Mat highlightImage;
    int blobNumber = blobCount(blobImage, highlightImage, true);
    cout << "The image has " << blobNumber<< " blobs"<<endl;
    namedWindow( "step5: blob count" , 0 );
    imshow("step5: blob count", blobImage);

    namedWindow( "step6: highlight blobs" , 0 );
    imshow("step6: highlight blobs", highlightImage);

    waitKey ( 0 ) ;
    return 0 ;
}

