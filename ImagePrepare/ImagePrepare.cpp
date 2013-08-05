// ImagePrepare.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

static bool readStringList(const string& filename, vector<string>& l )
{
	l.resize(0);
    FileStorage fs(filename, FileStorage::READ);
    if( !fs.isOpened() )
        return false;
    FileNode n = fs.getFirstTopLevelNode();
    if( n.type() != FileNode::SEQ )
        return false;
    FileNodeIterator it = n.begin(), it_end = n.end();
    for( ; it != it_end; ++it )
        l.push_back((string)*it);
    return true;
}


int main(int argc, char* argv[])
{

	// Raw data path:			C:/FaceDatabase/<subject>/nir .OR. visible
	// Calibration data path:	C:/FaceRecogniser/cam<id>/
	// Haarcascades:			C:/Haardata/
	// Landmark data:			C:/flandmark/
	// Save path:				C:/FaceRecogniser/<subject>/
	// image filename:			<subject>img<num>.png
	// CSV file stored in:		C:/FaceRecogniser/

	/* The subjects will have to be extracted manually */
	/* Need a yml or xml file to contain image names to be read*/

	const float delta = 0.05;
	const String face_cascade_name = "C:/HaarData/haarcascade_frontalface_alt.xml";
	const String eyes_cascade_name = "C:/HaarData/haarcascade_eye_tree_eyeglasses.xml";

	CascadeClassifier face_cascade;
	CascadeClassifier eyes_cascade;

	if (!face_cascade.load(face_cascade_name)) {std::cout << "Error loading face cascade\n"; return -1;}
	if (!eyes_cascade.load(eyes_cascade_name)) {std::cout << "Error loading eyes cascade\n"; return -1;}

	string subject, listname, num;

	string rawPath, calibPath, savePath, csvPath;


	string	csvFile = "C:/FaceRecogniser/csvTrain.txt";

	calibPath = "C:/FaceRecogniser/Cam1";
	listname = "images2read.xml";

	vector<string> imagelist; 
	ostringstream convert, convert2;   // stream used to convert int to string

	Mat M, D, timage;

	int w0, h0, w, h, dx, dy; 

	FileStorage fs;

	fs.open(calibPath + "/camMatrix1.yml", CV_STORAGE_READ);
	if(fs.isOpened())
	{
		fs["M"] >> M;
		fs["D"] >> D;
		fs.release();
	}
	else
	{
		std::cout << "Failed to open camera matrix file" << endl;
	}

		
	// These vectors hold the images, and face and eye locations
	Mat images[50], detect[50], rawimages[50], gimages[50];
	vector<Rect> faces;
	int eyes0[8], eyes[8];


	bool imageTemplate = false;

	ofstream outfile ( csvFile );

	for(int subjectID = 0; subjectID < 1; subjectID++)
	{

		//convert subjectID to string
		convert << subjectID;
		subject = "s" + convert.str();

		convert.str("");

		rawPath = "C:/FaceDataBase/" + subject + "/visible/";
		savePath = "C:/FaceRecogniser/" + subject + "/"; 

		bool ok = readStringList(rawPath + listname, imagelist);

		if (!ok || imagelist.empty())
		{
			std::cout << "Error opening " << listname << ", or the imagelist is empty\n" << endl;
			return -1;
		}

		int nimages = imagelist.size();

		std::cout << nimages << endl;

		for(int i=0; i < nimages; i++)
		{
			rawimages[i] = imread(rawPath + imagelist[i]); 
			undistort(rawimages[i], images[i], M, D);
			cvtColor(images[i], gimages[i], CV_BGR2GRAY);
			equalizeHist( gimages[i], timage );
			detect[i]= timage.clone(); 
		}


		//Detect face within the data (if no face detected -> skip image) 
		for( int i = 0; i < nimages; i++)
		{
			face_cascade.detectMultiScale( detect[i], faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50) );

			//if faces empty skip image
			if(faces.empty()) 
			{
				if(i==0) imageTemplate = true;
				std::cout << "No face detected in image:" << i << endl;
				continue;
			}

			//if more than one face detected skip image; could remove false faces by checking poisition & size of rect?
			if(faces.size() > 1)
			{
				if(i==0) imageTemplate = true;
				std::cout << "More than one face in image:" << i << endl;
				continue;
			}

			convert2 << i;
			num = convert2.str();
			csvPath = savePath + subject + "img"+ num +".png";

			convert2.str(""); 
		
			//Landmark the face -> eye data needed	
			landmark( images[i], faces[0], eyes);

			std::cout << "Landmarking done" << endl;

			if(i == 0 || imageTemplate)// this will be first image to register for a subject
			{
				imageTemplate = false; 

				//dx = floor(float(delta * faces[0].width)); 
				//dy = floor(float(delta * faces[0].height));

				//faces[0].x = eyes[10] - dx;
				//faces[0].width = eyes[12] - eyes[10] + 2*dx;

				//faces[0].height = faces[0].y + faces[0].height + dy - eyes[11];
				//faces[0].y = eyes[11] - dy;

				images[i] = images[i](faces[0]); 

				w0 = faces[0].width; 
				h0 = faces[0].height;

				for(int k = 0; k < 8; k += 2)
				{
					eyes0[k] = eyes[k] - faces[0].x;
					eyes0[k+1] = eyes[k+1] - faces[0].y; 

				}

				cv::imwrite(csvPath, images[i]);

				outfile << csvPath << ";" << subjectID << endl;

			}else // Any other image; match to the template size, align using eye data.
			{

				dx = floor(float(delta * faces[0].width)); 
				dy = floor(float(delta * faces[0].height));

				faces[0].x = eyes[10] - dx;
				faces[0].width = eyes[12] - eyes[10] + 2*dx;

				faces[0].height = faces[0].y + faces[0].height + dy - eyes[11];
				faces[0].y = eyes[11] - dy;

				images[i] = images[i](faces[0]); 

				//adjust eye location for face rectangle, and scale for resize.
				for(int k = 0; k < 8; k += 2)
				{
					eyes[k] = w0 * (eyes[k] - faces[0].x)/faces[0].width;
					eyes[k+1] = h0 * (eyes[k+1] - faces[0].y)/faces[0].height; 

				}

				cv::resize(images[i], images[i], Size(w0, h0), 1.0, 1.0, INTER_CUBIC);

				// for now skip transformations that will align data to template. 

				//save image, image path, and id label

				cv::imwrite(csvPath, images[i]);
				outfile << csvPath << ";" << subjectID << endl;
			}

		}
	
		std::cout << "Completed subject:" << subjectID << endl;
	}

	outfile.close();

	getchar();

	return 0;
}

static void landmark( Mat image, Rect rect_face, int eyes[] )
{
	//take clone of image to work with
	Mat cimage = image.clone();
	Mat greycimage;
	Mat cimagesplit[3]; 
	int blue[16], red[16], green[16], ave[16];
	string lmfname; 

	//separate out the blue, green, and red responses to NIR to provide three extra greyscale images
	// blue = 0, green = 1, red = 2  

	splitChannels(cimagesplit, cimage);

	cvtColor(cimage, greycimage, CV_BGR2GRAY);

	equalizeHist( greycimage, greycimage );
	equalizeHist( cimagesplit[0], cimagesplit[0] );
	equalizeHist( cimagesplit[1], cimagesplit[1] );
	equalizeHist( cimagesplit[2], cimagesplit[2] );

	//convert cv::Mat to IplImage type
	IplImage icimage = greycimage;
	IplImage iB = cimagesplit[0];
	IplImage iG = cimagesplit[1];
	IplImage iR = cimagesplit[2];

	//Apply landmarker
	const char* flandmarkModel = "C:/Projects/flandmarkData/flandmark_model.dat";

	FLANDMARK_Model * model = flandmark_init(flandmarkModel);

    if (model == 0)
    {
        printf("Structure model wasn't created. Corrupted or missing file flandmark_model.dat?\n");
        return;
    }

	int *bbox = (int*)malloc(4*sizeof(int));
    double *landmarks = (double*)malloc(2*model->data.options.M*sizeof(double));

	Mat LM(1,2*model->data.options.M, CV_16UC1);
		
	bbox[0] = rect_face.x;
	bbox[1] = rect_face.y;
	bbox[2] = rect_face.x + rect_face.width;
	bbox[3] = rect_face.y + rect_face.height;

	//detect landmarks using Blue channel
	flandmark_detect(&iB, bbox, model, landmarks);

	//extract blue channel data
	for (int i = 0; i < 2*model->data.options.M;i++)
    {
        blue[i]= int(landmarks[i]);
    }

	//detect landmarks using Green channel
	flandmark_detect(&iG, bbox, model, landmarks);
		
	//extract green channel data
	for (int i = 0; i < 2*model->data.options.M;i++)
    {
        green[i]= int(landmarks[i]);
    }

	//detect landmarks using Red channel
	flandmark_detect(&iR, bbox, model, landmarks);

	//extract red channel data
	for (int i = 0; i < 2*model->data.options.M;i++)
    {
        red[i]= int(landmarks[i]);
    }

	//detect landmarks using frame
	flandmark_detect(&icimage, bbox, model, landmarks);

	//average all four landmark data and store to LM
	for (int i = 0; i < 2*model->data.options.M;i++)
    {
        ave[i] = blue[i] + green[i] + red[i] + int(landmarks[i]);
		ave[i] *= 0.25;

		LM.at<uint16_t>(0,i) = ave[i];

    }

   // extract eye coordinates and pass as array back to main
	// eyes found at LM[2,3][4,5][10,11][12,13]

	for(int i = 0; i < 4; i++)
	{
		eyes[i] = LM.at<uint16_t>(0,i+2);
	}

	for(int i = 4; i < 8; i++)
	{
		eyes[i] = LM.at<uint16_t>(0,i+6);
	}

	cout << eyes[10] << endl;

	return;
}
//END OF LANDMARK --------------------------------------------------------------

static void splitChannels( Mat dst[], Mat& src )
{
	Mat R ( src.size(), CV_8UC1 );
    Mat G ( src.size(), CV_8UC1 );
    Mat B ( src.size(), CV_8UC1 );

	Mat out[]= { B, G, R };
	int from_to[] = { 0,0,  1,1,  2,2 };
	mixChannels( &src, 1, out, 3, from_to, 3);

	dst[0] = B;
	dst[1] = G;
	dst[2] = R;
}
//END OF FILE ------------------------------------------------------------------