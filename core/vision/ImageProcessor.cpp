#include <vision/ImageProcessor.h>
#include <vision/Classifier.h>
#include <vision/BeaconDetector.h>
#include <vision/Logging.h>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/LU>

ImageProcessor::ImageProcessor(VisionBlocks& vblocks, const ImageParams& iparams, Camera::Type camera) :
  vblocks_(vblocks), iparams_(iparams), camera_(camera), cmatrix_(iparams_, camera)
{
  enableCalibration_ = false;
  color_segmenter_ = std::make_unique<Classifier>(vblocks_, vparams_, iparams_, camera_);
  beacon_detector_ = std::make_unique<BeaconDetector>(DETECTOR_PASS_ARGS);
  calibration_ = std::make_unique<RobotCalibration>();
}

ImageProcessor::~ImageProcessor() {
}

void ImageProcessor::init(TextLogger* tl){
  textlogger = tl;
  vparams_.init();
  color_segmenter_->init(tl);
  beacon_detector_->init(tl);
}

unsigned char* ImageProcessor::getImg() {
  if(camera_ == Camera::TOP)
    return vblocks_.image->getImgTop();
  return vblocks_.image->getImgBottom();
}

//void ImageProcessor::saveImg(std::string filepath) {
//  cv::Mat mat;
//  int xstep_ = 1 << iparams_.defaultHorizontalStepScale;
//  int ystep_ = 1 << iparams_.defaultVerticalStepScale;
//  cv::resize(color_segmenter_->img_grayscale(), mat, cv::Size(), 1.0 / xstep_, 1.0 / ystep_, cv::INTER_NEAREST); 

//  cv::imwrite(filepath, mat);
//}

unsigned char* ImageProcessor::getSegImg(){
  if(camera_ == Camera::TOP)
    return vblocks_.robot_vision->getSegImgTop();
  return vblocks_.robot_vision->getSegImgBottom();
}

int* ImageProcessor::getGSegImg(){
  if(camera_ == Camera::TOP)
    return gseg_top;
  return gseg_bottom;
}

unsigned char* ImageProcessor::getColorTable(){
  return color_table_;
}

const CameraMatrix& ImageProcessor::getCameraMatrix(){
  return cmatrix_;
}

void ImageProcessor::updateTransform(){
  BodyPart::Part camera;
  if(camera_ == Camera::TOP)
    camera = BodyPart::top_camera;
  else
    camera = BodyPart::bottom_camera;

  Pose3D pcamera;
  if(enableCalibration_) {
    float joints[NUM_JOINTS], sensors[NUM_SENSORS], dimensions[RobotDimensions::NUM_DIMENSIONS];
    memcpy(joints, vblocks_.joint->values_.data(), NUM_JOINTS * sizeof(float));
    memcpy(sensors, vblocks_.sensor->values_.data(), NUM_SENSORS * sizeof(float));
    memcpy(dimensions, vblocks_.robot_info->dimensions_.values_, RobotDimensions::NUM_DIMENSIONS * sizeof(float));
    Pose3D *rel_parts = vblocks_.body_model->rel_parts_.data(), *abs_parts = vblocks_.body_model->abs_parts_.data();
    calibration_->applyJoints(joints);
    calibration_->applySensors(sensors);
    calibration_->applyDimensions(dimensions);
    ForwardKinematics::calculateRelativePose(joints, rel_parts, dimensions);
#ifdef TOOL
    Pose3D base = ForwardKinematics::calculateVirtualBase(calibration_->useLeft, rel_parts);
    ForwardKinematics::calculateAbsolutePose(base, rel_parts, abs_parts);
#else
    ForwardKinematics::calculateAbsolutePose(sensors, rel_parts, abs_parts);
#endif
    cmatrix_.setCalibration(*calibration_);
    pcamera = abs_parts[camera];
  }
  else pcamera = vblocks_.body_model->abs_parts_[camera];

  if(vblocks_.robot_state->WO_SELF == WO_TEAM_COACH) {
    auto self = vblocks_.world_object->objects_[vblocks_.robot_state->WO_SELF];
    pcamera.translation.z += self.height;
  }

  cmatrix_.updateCameraPose(pcamera);
}

bool ImageProcessor::isRawImageLoaded() {
  if(camera_ == Camera::TOP)
    return vblocks_.image->isLoaded();
  return vblocks_.image->isLoaded();
}

int ImageProcessor::getImageHeight() {
  return iparams_.height;
}

int ImageProcessor::getImageWidth() {
  return iparams_.width;
}

double ImageProcessor::getCurrentTime() {
  return vblocks_.frame_info->seconds_since_start;
}

void ImageProcessor::setCalibration(const RobotCalibration& calibration){
  *calibration_ = calibration;
}

void ImageProcessor::processFrame(){
  //cout<<" Process Frame !!!!" <<endl;
  if(vblocks_.robot_state->WO_SELF == WO_TEAM_COACH && camera_ == Camera::BOTTOM) return;
  tlog(30, "Process Frame camera %i", camera_);

  // Horizon calculation
  tlog(30, "Calculating horizon line");
  updateTransform();
  HorizonLine horizon = HorizonLine::generate(iparams_, cmatrix_, 20000);
  vblocks_.robot_vision->horizon = horizon;
  tlog(30, "Classifying Image: %i", camera_);
  if(!color_segmenter_->classifyImage(color_table_)) return;

  ballCandidates.clear();
  beaconCandidates.clear();
  goalCandidates.clear();
  excludeBeacon.clear();
  beaconPairs.clear();
  //omi();
  //return;
  buildBlobs();

  //processBeaconCandidates();
  processBallCandidates();
  processGoalCandidates();

  detectBall();
  //detectGoal();
  //beacon_detector_->findBeacons();
}

int djFind(int *dj, int u) {
  if(dj[u] == u) return u;
  return dj[u] = djFind(dj, dj[u]);
}

const int SZ = 320*240;
int root[SZ], dif[SZ], siz[SZ], rsiz[SZ];

int get_dif(unsigned char a[3], unsigned char b[3]) {
  int res = 0;
  int d = 0;
  for(int i=0; i<3; i++) {
    d = (int)a[i] - (int)b[i];
    res += d*d;
  }
  return sqrtf(res);
}

void ImageProcessor::omi() {
  long long start = clock();
  double tm;

  int W = iparams_.width, H = iparams_.height;
  //int root[W*H], dif[W*H], siz[W*H], rsiz[W*H];
  int total[W*H][3] = {};
  unsigned char yuv[W*H][3];

  const int dx[2] = {1, 0};
  const int dy[2] = {0, 1};
  const int K = 500;
  const int minSize = 10;
  const int dif_thres = 10;

  int wcnt[512+1] = {}, woff[512+1];
  long long tmpedge[W*H*2], edge[W*H*2];
  int tmpwei[W*H*2];

  auto *img = getImg();
  for(int y=0; y<H; y++) {
    for(int x=0; x<W; x++) {
      int i = y*W+x;
      int tx = x - (x&1);
      int idx = (y*W+tx) * 2;
      yuv[i][0] = img[idx+(x&1)*2];
      yuv[i][1] = img[idx+1];
      yuv[i][2] = img[idx+3];
    }
  }

  for(int i=0; i<W*H; i++) {
    root[i] = i;
    dif[i] = 0;
    siz[i] = 1;
    rsiz[i] = 1;
    for(int j=0; j<3; j++)
      total[i][j] = yuv[i][j];
  }

  int E = 0;
  for(int y=0; y<H; y++) {
    for(int x=0; x<W; x++) {
      int u = y*W+x;

      for(int i=0; i<2; i++) { 
        int ny = y + dy[i], nx = x + dx[i];
        if(ny >= H or nx >= W) continue;
        int v = ny*W+nx;

        int wei = get_dif(yuv[u], yuv[v]);
        tmpedge[E] = ((long long)u << 32) + v;
        tmpwei[E] = wei;
        wcnt[wei]++;
        E++;
      }
    }
  }

  woff[0] = 0;
  for(int i=1; i<=512; i++) {
    woff[i] = woff[i-1] + wcnt[i-1];
    wcnt[i-1] = 0;
  }

  for(int i=0; i<E; i++) {
    int w = tmpwei[i];
    edge[woff[w] + (wcnt[w]++)] = tmpedge[i];
  }

  tm = (clock() - start) / (double) CLOCKS_PER_SEC;
  tlog(30, "omi middle time: %.4fs", tm);

  for(int w=0; w<512; w++) {
    for(int i=woff[w]; i<woff[w+1]; i++) {
      int u = (edge[i] >> 32) & 0xffffffffll, v = edge[i] & 0xffffffffll;
      int pu = djFind(root, u), pv = djFind(root, v);
      if(pu == pv) continue;
      if(siz[pu] < siz[pv])
        swap(pu, pv);
      int mint = min(dif[pu] + K / siz[pu], dif[pv] + K / siz[pv]);
      unsigned char a[3], b[3];
      for(int i=0; i<3; i++) {
        a[i] = total[pu][i] / rsiz[pu];
        b[i] = total[pv][i] / rsiz[pv];
      }
      int cu = ColorTableMethods::yuv2color(color_table_, a[0], a[1], a[2]);
      int cv = ColorTableMethods::yuv2color(color_table_, b[0], b[1], b[2]);
      bool can_merge = (w <= mint) and (get_dif(a, b) < dif_thres or cu == cv);
      if(w <= mint or siz[pv] < minSize) {
        root[pv] = pu;
        siz[pu] += siz[pv];
        if(can_merge) {
          for(int j=0; j<3; j++)
            total[pu][j] += total[pv][j];
          dif[pu] = w;
          rsiz[pu] += rsiz[pv];
        }
      }
    }
  }

  for(int i=0; i<W*H; i++) {
    if(djFind(root, i) != i) continue;
    for(int j=0; j<3; j++)
      total[i][j] /= rsiz[i];
    for(int j=0; j<3; j++)
      assert(total[i][j] >= 0 and total[i][j] <= 255);
  }

  for(int i=0; i<W*H; i++) {
    int par = djFind(root, i);
    //getGSegImg()[i] = par;
    int c = ColorTableMethods::yuv2color(color_table_, total[par][0], total[par][1], total[par][2]);

    getGSegImg()[i] = (c << 24) + (total[par][2] << 16) + (total[par][1] << 8) + total[par][0];
  }

  tm = (clock() - start) / (double) CLOCKS_PER_SEC;
  tlog(30, "omi time: %.4fs", tm);
}

int Q[320*240];

void ImageProcessor::buildBlobs() {
  long long start = clock();
  int W = iparams_.width, H = iparams_.height;
  bool vis[H][W] = {};
  const int dx[4] = {1, -1, 0, 0};
  const int dy[4] = {0, 0, 1, -1};
  int ql = 0, qr = 0;
  int xcnt[W] = {};
  int xcnt_usedx[W] = {}, xcnt_tmp[W] = {};
  int xcnt_usedxcnt = 0;

  int blobCount = 0;

  for(int y=0; y<H; y++) {
    for(int x=0; x<W; x++) {
      if(vis[y][x]) continue;
      int blobIdx = blobCount;
      blobCount++;
      Q[qr++] = (y << 16) + x;
      vis[y][x] = true;

      int clr = getSegImg()[y * W + x];
      int blobSize = 0;
      int ymin = H, ymax = 0, xmin = W, xmax = 0;
      int avgx = 0, avgy = 0;

      //vector<pair<int,int>> pixels;

      while(ql != qr)
      {
        int y = (Q[ql] >> 16) & 0xffff, x = Q[ql] & 0xffff;
        ql++;

        //getGSegImg()[y * W + x] = blobIdx;

        //pixels.push_back({x, y});
        if(!xcnt[x]) {
          xcnt_usedx[xcnt_usedxcnt++] = x;
        }
        xcnt[x]++;

        ymin = min(ymin, y);
        ymax = max(ymax, y);
        xmin = min(xmin, x);
        xmax = max(xmax, x);
        avgx += x;
        avgy += y;
        blobSize++;

        for(int i=0; i<4; i++)
        {
          int ny = y + dy[i], nx = x + dx[i];
          if(ny < 0 or ny >= H or nx < 0 or nx >= W) continue;
          if(vis[ny][nx]) continue;
          if(getSegImg()[ny * W + nx] != clr) continue;
          Q[qr++] = (ny << 16) + nx;
          vis[ny][nx] = true;
        }
      }
      avgx /= blobSize;
      avgy /= blobSize;

      int boundingBoxArea = (ymax - ymin + 1) * (xmax - xmin + 1);
      double density = blobSize / (double) boundingBoxArea;

      //ball
      if(blobSize >= 10 and clr == c_ORANGE) {
        auto bc = BallCandidate();
        bc.centerX = (xmin + xmax) / 2;
        bc.centerY = (ymin + ymax) / 2;
        bc.width = xmax - xmin + 1;
        bc.height = ymax - ymin + 1;
        bc.radius = (bc.width + bc.height) / 4;
        bc.valid = true;

        double aspect_ratio = bc.width / bc.height;

        if(0.5 < density and 0.7 < aspect_ratio and aspect_ratio < 1.4) { 
          ballCandidates.push_back(bc);
        }
      }

      //if(blobSize >= 15)
        //tlog(30, "Color %d Size %d [%d~%d] x [%d~%d] Density %.3f", clr, blobSize, xmin, xmax, ymin, ymax, density);

      //beacon
      if(0)
      if(blobSize >= 15 and (clr == c_YELLOW or clr == c_BLUE or clr == c_PINK)) {
        auto bc = BallCandidate();
        //bc.centerX = (xmin + xmax) / 2;
        //bc.centerY = (ymin + ymax) / 2;
        bc.centerX = avgx;
        bc.centerY = avgy;
        bc.width = xmax - xmin + 1;
        bc.height = ymax - ymin + 1;
        bc.radius = (bc.width + bc.height) / 4;
        bc.valid = true;
        bc.clr = clr;
        double aspect_ratio = bc.width / bc.height;
        bc.ar = aspect_ratio;

        bc.index = blobIdx;


        //if(0.4 < density and 0.3 < aspect_ratio and aspect_ratio < 3) { 
        if(0.4 < density) { 
          beaconCandidates.push_back(bc);
        }
      }




      //goal
      //if(blobSize >= 2000 and clr == c_BLUE) {
      //if(0)
      if(blobSize >= 500 and clr == c_BLUE) {

        auto bc = BallCandidate();
        bc.centerX = (xmin + xmax) / 2;
        bc.centerY = (ymin + ymax) / 2;
        bc.width = xmax - xmin + 1;
        bc.height = ymax - ymin + 1;

        bc.radius = (bc.width + bc.height) / 4;
        bc.valid = true;
        bc.clr = clr;
        double aspect_ratio = bc.width / bc.height;
        bc.ar = aspect_ratio;

        bc.index = blobIdx;


        if(density >= 0.4 and aspect_ratio > 0.2 and aspect_ratio < 5) {
          for(int i=0; i<xcnt_usedxcnt; i++)
            xcnt_tmp[i] = xcnt[xcnt_usedx[i]];
          int pos = xcnt_usedxcnt / 2;
          nth_element(xcnt_tmp, xcnt_tmp + pos, xcnt_tmp + xcnt_usedxcnt);
          int midh = xcnt_tmp[pos] / 2;

          //sort(pixels.begin(), pixels.end());
          //vector<int> counter;
          //int curx = -1, sumx = -1;
          //for(int p=0;p<pixels.size();p++){
            //int haox = pixels[p].first;
            //int haoy = pixels[p].second;

            //if(haox!=curx){
              //if(sumx!=-1) counter.push_back(sumx);
              //curx = haox;
              //sumx = 1;
            //}
            //else
              //sumx += 1;
          //}
          //sort(counter.begin(), counter.end());

          //double midh = 0;
          //if(!counter.empty())
            //midh = counter.at(counter.size()/2)/2;
          bc.midh = midh;
          goalCandidates.push_back(bc);
          //cout<<"GOAL GOOD "<<xcnt_usedxcnt<<" Diff Xs, Mid Height "<<zmidh<<" Orig "<<midh<<endl;
        }
      }

      // Clear xcnt
      for (int i=0; i<xcnt_usedxcnt; i++)
        xcnt[xcnt_usedx[i]] = 0;
      xcnt_usedxcnt = 0;
    }
  }




      //cout<<cmatrix_.cameraPosition_ << endl;
      //cout<<cmatrix_.worldToCam_<<endl<<endl;

      //cout<<cmatrix_.camToWorld_<<endl;
      //Position p = cmatrix_.getWorldPosition(274, 199, 50);
      //Position p = cmatrix_.getWorldPosition(160, 180, 50);
      //cout<<cmatrix_.bearing(p)<<" "<<cmatrix_.elevation(p)<<" "<<cmatrix_.groundDistance(p)<<endl;



      double tm = (clock() - start) / (double) CLOCKS_PER_SEC;
      tlog(30, "buildBlob time: %.4fs", tm);
    }


    void ImageProcessor::processGoalCandidates() {


      for(auto bc: goalCandidates) {
        if(find(excludeBeacon.begin(), excludeBeacon.end(), bc.index) != excludeBeacon.end()) {
          //cout<<bc.index<<endl; 
          continue;
        }


        double midh = bc.midh;
        WorldObject* goal = &vblocks_.world_object->objects_[WO_OWN_GOAL];
        goal->imageCenterX = bc.centerX;
        goal->imageCenterY = bc.centerY;
        gw = bc.width;
        gh = bc.height;
        Position p = cmatrix_.getWorldPosition(bc.centerX, bc.centerY, 255);
        goal->visionBearing = cmatrix_.bearing(p);
        goal->visionElevation = cmatrix_.elevation(p);
        goal->visionDistance = cmatrix_.groundDistance(p);
        goal->fromTopCamera = camera_ == Camera::TOP;
        goal->seen = true;


        int x1 = bc.centerX, y1 = bc.centerY-midh, x2 = bc.centerX, y2 = bc.centerY+midh;
        Eigen::Vector3f v1, v2;
        v1 << x1, y1, 1;
        v2 << x2, y2, 1;
        auto Kinv = cmatrix_.cameraCalibration_.inverse();
        auto u1 = Kinv * v1, u2 = Kinv * v2;
        auto c12 = (u1.dot(u2)) / (u1.norm() * u2.norm());
        double theta = acos(c12);

        double h0 = cmatrix_.cameraPosition_[2];
        double dh1 = h0 - 510, dh2 = h0;

        double lb = 0, rb = 10000;
        for(int i=0; i<15; i++) {
          double mb = (lb + rb) / 2;
          double val = atan2(dh2, mb) - atan2(dh1, mb);
          if(val < theta)
            rb = mb;
          else
            lb = mb;
        }

        double dis = lb;
        double disrat = dis / goal->visionDistance;
        goal->distance = dis;

        if(dis>5000 or disrat < 0.3 or disrat > 3) goal->seen = false;
      }

      /*
         p = cmatrix_.getWorldPosition(centerX, centerY, 255);
         goal->visionBearing = cmatrix_.bearing(p);
         goal->visionElevation = cmatrix_.elevation(p);
         goal->visionDistance = cmatrix_.groundDistance(p);
         */



    }


    void ImageProcessor::processBeaconCandidates() {
      int length = beaconCandidates.size();
      if (length>0){
        bool exist[length] = {false};
        for(int i = 0;i<length;i++){
          for(int j = i+1; j<length;j++){
            //if(i == j) continue;
            BallCandidate& b1 = beaconCandidates[i];
            BallCandidate& b2 = beaconCandidates[j];
            if(b1.clr == b2.clr) continue;
            //cout<<b1.centerX<<","<<b1.centerY<<" "<<b1.width<<"x"<<b1.height<<endl;
            //cout<<b2.centerX<<","<<b2.centerY<<" "<<b2.width<<"x"<<b2.height<<endl;
            if(b1.width/b2.width<0.5 or b1.width/b2.width>2 or b1.height/b2.height<0.5 or b1.height/b2.height>2) continue;
            if( abs(b1.centerX - b2.centerX)> 0.5*(b1.radius + b2.radius)) continue;

            int dx = b1.centerX - b2.centerX, dy = b1.centerY - b2.centerY;
            double image_dis = hypot(dx, dy);
            if( image_dis > 1.5*(b1.height + b2.height)/2 or image_dis < 0.3*(b1.height+b2.height)/2) continue;


            //BallCandidate bc;
            //if(b1.centerY > b2.centerY) bc = b1; else bc = b2;
            int total = 0, white = 0;
            int step = max(1, (int)b2.radius / 10);
            int wx = 2 * b2.centerX - b1.centerX, wy = 2 * b2.centerY - b1.centerY;
            int amix = b2.width / 3, amiy = b2.height / 3;
            for(int y=wy-amiy; y<=wy+amiy; y+=step) {
              for(int x=wx-amix; x<=wx+amix; x+=step) {
                total++;
                if (y < 0 or y >= iparams_.height or x < 0 or x >= iparams_.width)
                  continue;
                if(getSegImg()[y * iparams_.width + x] == c_WHITE or getSegImg()[y * iparams_.width + x] == c_ROBOT_WHITE)
                  white++;
              }
            }
            tlog(30,"hahaha %.2f",(double)white/total);
            if((double)white/total<0.35) continue;
            /////////////////////////////////////
            //remember to test false positive
            /////////////////////////////////////
            int total1 = 0, blue1 = 0, yellow1 = 0, pink1 = 0;
            step = max(1, (int)b1.radius / 10);
            wx = 2 * b1.centerX - b2.centerX, wy = 2 * b1.centerY - b2.centerY;
            amix = b1.width / 3, amiy = b1.height / 3;
            for(int y=wy; y<=wy+amiy; y+=step) {
              for(int x=wx-amix; x<=wx+amix; x+=step) {
                total1++;
                if (y < 0 or y >= iparams_.height or x < 0 or x >= iparams_.width)
                  continue;
                if(getSegImg()[y * iparams_.width + x] == c_PINK) pink1++;
                else if(getSegImg()[y * iparams_.width + x] == c_BLUE) blue1++;
                else if(getSegImg()[y * iparams_.width + x] == c_YELLOW) yellow1++;
              }
            }
            
            int total2 = 0, blue2 = 0, yellow2 = 0, pink2 = 0;
            step = max(1, (int)b1.radius / 10);
            wx = 2 * b1.centerX - b2.centerX, wy = 2 * b1.centerY - b2.centerY;
            amix = b1.width / 3, amiy = b1.height / 3;
            for(int y=wy-2*amiy; y<=wy+1*amiy; y+=step) {
              for(int x=wx-2*amix; x<=wx+2*amix; x+=step) {
                total2++;
                if (y < 0 or y >= iparams_.height or x < 0 or x >= iparams_.width)
                  continue;
                if(getSegImg()[y * iparams_.width + x] == c_PINK) pink2++;
                else if(getSegImg()[y * iparams_.width + x] == c_BLUE) blue2++;
                else if(getSegImg()[y * iparams_.width + x] == c_YELLOW) yellow2++;
              }
            }
            double th1 = 0.5;
            double th2 = 0.5;
            if(( (double)blue1/total1>th1 and (double)blue2/total2<th2) or ((double)yellow1/total1>th1 and (double)yellow2/total2<th2) or ((double)pink1/total1>th1 and (double)pink2/total2< th2)) continue;
            /////////////////////////////////////
            //remember to test false positive
            /////////////////////////////////////
            




            exist[i] = true; 
            exist[j] = true; 
            beaconPairs.push_back({i, j});
          }
        }

        vector<pair<int,int> > new_beaconPairs;
        for(int i = 0;i<beaconPairs.size();i++){
          int b1 = beaconPairs[i].first, b2 = beaconPairs[i].second;
          if(exist[b1] and exist[b2]) new_beaconPairs.push_back(make_pair(b1,b2));
        }

        // 好三三


        map<WorldObjectType,vector<int>> beacons = {
          /*
             { WO_BEACON_YELLOW_BLUE, { 24, 15} }
             */
        };

        map<WorldObjectType,int> heights = {
          { WO_BEACON_YELLOW_BLUE, 300 },
          { WO_BEACON_BLUE_YELLOW, 300 },
          { WO_BEACON_YELLOW_PINK, 200 },
          { WO_BEACON_PINK_YELLOW, 200 },
          { WO_BEACON_BLUE_PINK, 200 },
          { WO_BEACON_PINK_BLUE, 200 }
        };


        for(int i = 0;i<new_beaconPairs.size();i++){
          BallCandidate& b1 = beaconCandidates[new_beaconPairs[i].first];
          BallCandidate& b2 = beaconCandidates[new_beaconPairs[i].second];
          tlog(30, "%.2f %.2f", b1.ar, b2.ar);
          bool occlude = (b1.ar<0.75 or b2.ar<0.75);
          //ballCandidates.push_back(BallCandidate(beaconCandidates[new_beaconPairs[i].first]));
          //ballCandidates.push_back(BallCandidate(beaconCandidates[new_beaconPairs[i].second]));
          WorldObjectType bt;
          if(b1.clr==c_YELLOW and b2.clr==c_BLUE) bt = WO_BEACON_YELLOW_BLUE;
          else if(b1.clr==c_BLUE and b2.clr==c_YELLOW) bt = WO_BEACON_BLUE_YELLOW;
          else if(b1.clr==c_YELLOW and b2.clr==c_PINK) bt = WO_BEACON_YELLOW_PINK;
          else if(b1.clr==c_PINK and b2.clr==c_YELLOW) bt = WO_BEACON_PINK_YELLOW;
          else if(b1.clr==c_BLUE and b2.clr==c_PINK) bt = WO_BEACON_BLUE_PINK;
          else if(b1.clr==c_PINK and b2.clr==c_BLUE) bt = WO_BEACON_PINK_BLUE;

          beacons[bt] = {
            (int)(b1.centerX + b2.centerX)/2, (int)(b1.centerY + b2.centerY)/2,
            (int)b1.centerX, (int)b1.centerY, (int)b2.centerX, (int)b2.centerY, occlude
          };

          //if(b1.clr==c_BLUE) cout<<b1.index<<endl;
          //if(b2.clr==c_BLUE) cout<<b2.index<<endl;
          if(b1.clr==c_BLUE) excludeBeacon.push_back(b1.index);
          if(b2.clr==c_BLUE) excludeBeacon.push_back(b2.index);
        }


        for(auto beacon : beacons) {
          auto& object = vblocks_.world_object->objects_[beacon.first];
          auto box = beacon.second;
          object.imageCenterX = box[0];
          object.imageCenterY = box[1];
          auto position = cmatrix_.getWorldPosition(object.imageCenterX, object.imageCenterY, heights[beacon.first]);
          object.visionDistance = cmatrix_.groundDistance(position);
          object.visionBearing = cmatrix_.bearing(position);
          object.seen = true;
          object.fromTopCamera = camera_ == Camera::TOP;

          int x1 = box[2], y1 = box[3], x2 = box[4], y2 = box[5];
          Eigen::Vector3f v1, v2;
          v1 << x1, y1, 1;
          v2 << x2, y2, 1;
          auto Kinv = cmatrix_.cameraCalibration_.inverse();
          auto u1 = Kinv * v1, u2 = Kinv * v2;
          auto c12 = (u1.dot(u2)) / (u1.norm() * u2.norm());
          double theta = acos(c12);

          double h0 = cmatrix_.cameraPosition_[2];
          double dh1 = h0 - (heights[beacon.first] + 50), dh2 = h0 - (heights[beacon.first] - 50);

          double lb = 0, rb = 10000;
          for(int i=0; i<15; i++) {
            double mb = (lb + rb) / 2;
            double val = atan2(dh2, mb) - atan2(dh1, mb);
            if(val < theta)
              rb = mb;
            else
              lb = mb;
          }

          double dis = lb;
          object.distance = dis;
          object.occlude = box[6];
          tlog(30,"%d",box[6]);


          tlog(30, "saw %s at (%i,%i) with calculated distance %2.4f", getName(beacon.first), object.imageCenterX, object.imageCenterY, object.visionDistance);
        }
      }
    }


    void ImageProcessor::processBallCandidates() {
      vector<BallCandidate> new_cands;

      double ball_rad = 25;
      for(auto bc: ballCandidates) {
        Position p = cmatrix_.getWorldPosition(bc.centerX, bc.centerY, ball_rad);
        bc.relPosition = p;
        int distance = cmatrix_.groundDistance(p);


        Eigen::Vector3f v1, v2, v3;
        v1 << bc.centerX, bc.centerY, 1;
        v2 << bc.centerX + bc.radius, bc.centerY, 1;
        v3 << bc.centerX, bc.centerY + bc.radius, 1;
        auto Kinv = cmatrix_.cameraCalibration_.inverse();
        auto u1 = Kinv * v1, u2 = Kinv * v2, u3 = Kinv * v3;
        auto c12 = (u1.dot(u2)) / (u1.norm() * u2.norm());
        auto c13 = (u1.dot(u3)) / (u1.norm() * u3.norm());
        double theta = (acos(c12) + acos(c13)) / 2;

        double h0 = cmatrix_.cameraPosition_[2] - ball_rad;
        double straightDistance = ball_rad / sin(theta);
        bc.groundDistance = sqrt(max(straightDistance * straightDistance - h0 * h0, 2500.));

        int xmin = 10000, xmax = -10000, ymin = 10000, ymax = -10000;
        for(int i=0; i<12; i++) {
          for(int j=-3; j<=3; j++) {
            double theta = j * M_PI / 6, phi = i * M_PI / 6;
            double x = p.x + ball_rad * cos(theta) * cos(phi);
            double y = p.y + ball_rad * cos(theta) * sin(phi);
            double z = p.z + ball_rad * sin(theta);
            auto res = cmatrix_.getImageCoordinates(x, y, z);
            xmin = min(xmin, res.x);
            xmax = max(xmax, res.x);
            ymin = min(ymin, res.y);
            ymax = max(ymax, res.y);
          }
        }

        double goodrad = ((xmax - xmin) + (ymax - ymin)) / 4.0;
        tlog(30, "BC (%d, %d) Rad %d GoodRad %d Dist %d", bc.centerX, bc.centerY, bc.radius, goodrad, distance);
        //printf("BC (%d, %d) Rad %d GoodRad %d Dist %d", (int)bc.centerX, (int)bc.centerY, (int)bc.radius, (int)goodrad, (int)distance);

        if(goodrad / bc.radius > 2 or bc.radius / goodrad > 2)
          continue;

        // Neighbor test : the neighborhood of the ball shouldn't be too orange
        int total = 0, orange = 0;
        int step = max(1, (int)bc.radius / 5);
        for(int y=bc.centerY-bc.height; y<=bc.centerY+bc.height; y+=step) {
          for(int x=bc.centerX-bc.width; x<=bc.centerX+bc.width; x+=step) {
            total++;
            if (y < 0 or y >= iparams_.height or x < 0 or x >= iparams_.width)
              continue;
            auto c = getSegImg()[y * iparams_.width + x];
            if(c == c_ORANGE or c == c_PINK)
              orange++;
          }
        }

        double orange_ratio = (double) orange / (total + 1);
        tlog(30, "Neighbor orange %.3f", orange_ratio);

        if (orange_ratio > 0.3)
          continue;

        new_cands.push_back(bc);
      }

      if(!new_cands.empty()) {
        int max_idx = -1;
        for(int i=0; i<(int)new_cands.size(); i++) {
          if(max_idx == -1 or (new_cands[i].radius > new_cands[max_idx].radius))
            max_idx = i;
        }
        swap(new_cands[0], new_cands[max_idx]);
      }

      ballCandidates = new_cands;
    }

    void ImageProcessor::detectBall() {
      int imageX, imageY;
      if (ballCandidates.empty()) return;
      WorldObject* ball = &vblocks_.world_object->objects_[WO_BALL];
      BallCandidate &bc = ballCandidates[0];
      if(ball and ball->seen and (ball->radius > bc.radius)) return;

      ball->imageCenterX = bc.centerX;
      ball->imageCenterY = bc.centerY;

      Position p = bc.relPosition;
      //Position p = cmatrix_.getWorldPosition(imageX, imageY, 50);
      ball->visionBearing = cmatrix_.bearing(p);
      ball->visionElevation = cmatrix_.elevation(p);
      ball->visionDistance = cmatrix_.groundDistance(p);
      ball->fromTopCamera = camera_ == Camera::TOP;
      ball->radius = bc.radius;

      tlog(30, "Ball detected");
      //cout<<"Ball Hao123 "<<bc.centerX<<" , "<<bc.centerY<<endl;

      ball->seen = true;
    }

    void ImageProcessor::detectGoal() {
      //if(camera_ == Camera::BOTTOM)
        //return;

      //int imageX, imageY;
      //double area;
      //findGoal(imageX, imageY, area);
      //if (imageX == -1) return; // function defined elsewhere that fills in imageX, imageY by reference
      //WorldObject* goal = &vblocks_.world_object->objects_[WO_OWN_GOAL];

      //goal->imageCenterX = imageX;
      //goal->imageCenterY = imageY;

      //Position p = cmatrix_.getWorldPosition(imageX, imageY, 280);
      //goal->visionBearing = cmatrix_.bearing(p);
      //goal->visionElevation = cmatrix_.elevation(p);
      ////goal->visionDistance = area;
      //goal->visionDistance = cmatrix_.groundDistance(p);
      //goal->fromTopCamera = camera_ == Camera::TOP;

      //goal->seen = true;
    }

    void ImageProcessor::findGoal(int& imageX, int& imageY, double& area) {
      //imageX = imageY = -1;

      //int total = 0;
      //int sumx = 0, sumy = 0;
      //int step = iparams_.width / 320; 

      //int W = iparams_.width / step, H = iparams_.height / step;

      //int xcnt[W] = {};
      //int ycnt[H] = {};

      //// Process from left to right
      //for(int x = 0; x < iparams_.width; x+=step) {
        //// Process from top to bottom
        //for(int y = 0; y < iparams_.height; y+=step) {
          //// Retrieve the segmented color of the pixel at (x,y)
          //auto c = getSegImg()[y * iparams_.width + x];
          //if(c == c_BLUE)
          //{
            //total++;
            //sumx += x;
            //sumy += y;
            //xcnt[x/step]++;
            //ycnt[y/step]++;
          //}
        //}
      //}

      //double ratio = (double) total / ((iparams_.width/step) * (iparams_.height/step));
      //if (ratio > 0.02)
      //{
        //imageX = sumx / total;
        //imageY = sumy / total;

        //int xmin = -1, xmax = -1, ymin = -1, ymax = -1;
        //int lth = 0.05 * total, rth = 0.95 * total;
        //int cur = 0;
        //for(int i=0; i<W; i++)
        //{
          //int ncur = cur + xcnt[i];
          //if(cur < lth and ncur >= lth)
            //xmin = i;
          //if(cur < rth and ncur >= rth)
            //xmax = i;
          //cur = ncur;
        //}
        //cur = 0;
        //for(int i=0; i<H; i++)
        //{
          //int ncur = cur + ycnt[i];
          //if(cur < lth and ncur >= lth)
            //ymin = i;
          //if(cur < rth and ncur >= rth)
            //ymax = i;
          //cur = ncur;
        //}

        ////area = ratio;
        //area = (double)((xmax - xmin) * (ymax - ymin)) / (W*H);
        ////cout<<"["<<xmin*100/W<<"~"<<xmax*100/W<<"] x ["<<ymin*100/H<<"~"<<ymax*100/H<<"]"<<endl;
      //}
    }


    int ImageProcessor::getTeamColor() {
      return vblocks_.robot_state->team_;
    }

    void ImageProcessor::SetColorTable(unsigned char* table) {
      color_table_ = table;
    }

    float ImageProcessor::getHeadChange() const {
      if (vblocks_.joint == NULL)
        return 0;
      return vblocks_.joint->getJointDelta(HeadPan);
    }

    std::vector<BallCandidate*> ImageProcessor::getBallCandidates() {
      auto ret = std::vector<BallCandidate*>();
      for (auto &bc: ballCandidates)
        ret.push_back(&bc);
      return ret;
    }

    BallCandidate* ImageProcessor::getBestBallCandidate() {
      if (ballCandidates.empty())
        return NULL;
      else
        return &ballCandidates[0];
    }

    void ImageProcessor::enableCalibration(bool value) {
      enableCalibration_ = value;
    }

    bool ImageProcessor::isImageLoaded() {
      return vblocks_.image->isLoaded();
    }
