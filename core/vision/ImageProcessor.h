#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H

#include <kinematics/ForwardKinematics.h>
#include <common/RobotDimensions.h>
#include <common/Profiling.h>
#include <memory/TextLogger.h>
#include <vision/CameraMatrix.h>
#include <vision/VisionBlocks.h>
#include <common/RobotInfo.h>
#include <common/RobotCalibration.h>
#include <vision/structures/BallCandidate.h>
#include <math/Pose3D.h>
#include <vision/structures/VisionParams.h>

class BallDetector;
class Classifier;
class BeaconDetector;

/// @ingroup vision
class ImageProcessor {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW  
    ImageProcessor(VisionBlocks& vblocks, const ImageParams& iparams, Camera::Type camera);
    ~ImageProcessor();
    void processFrame();
    void init(TextLogger*);
    void SetColorTable(unsigned char*);
    std::unique_ptr<BeaconDetector> beacon_detector_;
    std::unique_ptr<Classifier> color_segmenter_;
    unsigned char* getImg();
    unsigned char* getSegImg();
    int* getGSegImg();
    unsigned char* getColorTable();
    bool isRawImageLoaded();
    int getImageHeight();
    int getImageWidth();
    const ImageParams& getImageParams() const { return iparams_; }
    const CameraMatrix& getCameraMatrix();
    void setCalibration(const RobotCalibration& calibration);
    void enableCalibration(bool value);
    void updateTransform();
    std::vector<BallCandidate*> getBallCandidates();
    BallCandidate* getBestBallCandidate();
    bool isImageLoaded();

    void omi();
    void buildBlobs();
    void processBallCandidates();
    void processBeaconCandidates();
    void processGoalCandidates();

    void detectBall();
    void findBall(int& imageX, int& imageY);
    void detectGoal();
    void findGoal(int& imageX, int& imageY, double& area);
    
    int gw, gh;

  private:
    int getTeamColor();
    double getCurrentTime();

    VisionBlocks& vblocks_;
    const ImageParams& iparams_;
    Camera::Type camera_;
    CameraMatrix cmatrix_;
    
    VisionParams vparams_;
    unsigned char* color_table_;
    TextLogger* textlogger;

    float getHeadPan() const;
    float getHeadTilt() const;
    float getHeadChange() const;
    
    std::unique_ptr<RobotCalibration> calibration_;
    bool enableCalibration_;

    //void saveImg(std::string filepath);
    int topFrameCounter_ = 0;
    int bottomFrameCounter_ = 0;

    std::vector<BallCandidate> ballCandidates;
    std::vector<BallCandidate> beaconCandidates;
    std::vector<BallCandidate> goalCandidates;
    std::vector<pair<int,int>> beaconPairs;
    std::vector<int> excludeBeacon;

    int gseg_top[100000], gseg_bottom[100000];


};

#endif
