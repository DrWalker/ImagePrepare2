// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include "targetver.h"
#include "opencv2\opencv.hpp"

#include "flandmark_detector.h"


#include <stdio.h>
#include <tchar.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdlib.h>

using namespace std; 
using namespace cv;

static void landmark( Mat image, Rect rect_face, int eyes[] );
static void splitChannels( Mat dst[], Mat& src );

// TODO: reference additional headers your program requires here
