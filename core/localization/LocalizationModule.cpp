#include <localization/LocalizationModule.h>
#include <memory/WorldObjectBlock.h>
#include <memory/LocalizationBlock.h>
#include <memory/GameStateBlock.h>
#include <memory/RobotStateBlock.h>
#include <localization/ParticleFilter.h>
#include <localization/Logging.h>

// Boilerplate
LocalizationModule::LocalizationModule() : tlogger_(textlogger), pfilter_(new ParticleFilter(cache_, tlogger_)), \
                                             kfilter_(new ExtKalmanFilter(tlogger_)) {
}

LocalizationModule::~LocalizationModule() {
  delete pfilter_;
  delete kfilter_;
}

// Boilerplate
void LocalizationModule::specifyMemoryDependency() {
  requiresMemoryBlock("world_objects");
  requiresMemoryBlock("localization");
  requiresMemoryBlock("vision_frame_info");
  requiresMemoryBlock("robot_state");
  requiresMemoryBlock("game_state");
  requiresMemoryBlock("vision_odometry");
}

// Boilerplate
void LocalizationModule::specifyMemoryBlocks() {
  getOrAddMemoryBlock(cache_.world_object,"world_objects");
  getOrAddMemoryBlock(cache_.localization_mem,"localization");
  getOrAddMemoryBlock(cache_.frame_info,"vision_frame_info");
  getOrAddMemoryBlock(cache_.robot_state,"robot_state");
  getOrAddMemoryBlock(cache_.game_state,"game_state");
  getOrAddMemoryBlock(cache_.odometry,"vision_odometry");
}


// Load params that are defined in cfglocalization.py
void LocalizationModule::loadParams(LocalizationParams params) {
  params_ = params;
  printf("Loaded localization params for %s\n", params_.behavior.c_str());
}

// Perform startup initialization such as allocating memory
void LocalizationModule::initSpecificModule() {
  reInit();
}

// Initialize the localization module based on data from the LocalizationBlock
void LocalizationModule::initFromMemory() {
  reInit();
}

// Initialize the localization module based on data from the WorldObjectBlock
void LocalizationModule::initFromWorld() {
  reInit();
  auto& self = cache_.world_object->objects_[cache_.robot_state->WO_SELF];
  pfilter_->init(self.loc, self.orientation);
  kfilter_->reset();
}

// Reinitialize from scratch
void LocalizationModule::reInit() {
  cache_.localization_mem->player_ = Point2D(-1250,0);
  cache_.localization_mem->state = decltype(cache_.localization_mem->state)::Zero();
  cache_.localization_mem->covariance = decltype(cache_.localization_mem->covariance)::Identity();
  pfilter_->init(cache_.localization_mem->player_, 0.0f);
  kfilter_->reset();
  cache_.world_object->objects_[cache_.robot_state->WO_SELF].orientation = 0.;
  last_frame_time = clock();
}

void LocalizationModule::moveBall(const Point2D& position) {
  // Optional: This method is called when the player is moved within the localization
  // simulator window.
}

void LocalizationModule::movePlayer(const Point2D& position, float orientation) {
  // Optional: This method is called when the player is moved within the localization
  // simulator window.
}

void LocalizationModule::processFrame() {
  auto& ball = cache_.world_object->objects_[WO_BALL];
  auto& self = cache_.world_object->objects_[cache_.robot_state->WO_SELF];

  // Retrieve the robot's current location from localization memory
  // and store it back into world objects
  auto sloc = cache_.localization_mem->player_;
  self.loc = sloc;
  tlog(30, "Self: (%f, %f) Orien %f", sloc.x, sloc.y, self.orientation);

  double delta_t = (clock() - last_frame_time) / (double)CLOCKS_PER_SEC;
  tlog(30, "dt = %.3f sec", delta_t);
  last_frame_time = clock();
  kfilter_->motionUpdate({}, delta_t);

  if(ball.seen) {
    last_ball_seen = clock();

    // Compute the relative position of the ball from vision readings
    auto relBall = Point2D::getPointFromPolar(ball.visionDistance, ball.visionBearing);
    tlog(30, "RelBall: (%f, %f) VDis %.0f Dir %.0f", relBall.x, relBall.y, ball.visionDistance, ball.visionBearing * RAD_T_DEG);

    VectorOd obs;
    obs << relBall.x, relBall.y;

    double logLikelihood = kfilter_->getObsLogLikelihood(obs);
    tlog(30, "log(obs) = %f", logLikelihood);
    
    kfilter_->measureUpdate(obs);
  }
  if(clock() - last_ball_seen >= 2.0 * CLOCKS_PER_SEC)
    kfilter_->reset();

  MatrixOd cov;
  VectorOd kf_obs = kfilter_->getObs(&cov);
  VectorSd kf_state = kfilter_->getMean();
  MatrixSd kf_cov = kfilter_->getCov();

  stringstream ss;
  string s;
  ss << fixed<<setprecision(0);
  ss << kf_state.transpose();
  s = ss.str();
  tlog(30, "State: [%s]", s.c_str());
  ss.str("");
  ss << kf_cov;
  s = ss.str();
  tlog(30, "State: [\n%s\n]", s.c_str());

  auto kf_relBall = Point2D(kf_obs(STATE_X), kf_obs(STATE_Y));

  // Compute the global position of the ball based on our assumed position and orientation
  auto globalBall = kf_relBall.relativeToGlobal(self.loc, self.orientation);
  auto globalCov = cov; // should rotate, but now self.orientation == 0

  // Update the ball in the WorldObject block so that it can be accessed in python
  ball.loc = globalBall;
  tlog(30, "Ball: (%f, %f)", ball.loc.x, ball.loc.y);
  tlog(30, "Cov: [ %f, %f ]", cov(0, 0), cov(0, 1));
  tlog(30, "     [ %f, %f ]", cov(1, 0), cov(1, 1));
  ball.distance = kf_relBall.getMagnitude();
  ball.bearing = kf_relBall.getDirection();
  ball.absVel = Point2D(kf_state(STATE_VELX), kf_state(STATE_VELY));
  ball.sd = ball.loc + (ball.absVel / TRANS_DAMP_K); // Predicted stop position

  cout<<fixed<<setprecision(0);
  cout<<"Ball "<<kf_state.transpose()<<" Target "<<ball.sd<<" dt "<<delta_t*1000<<endl;

  // Update the localization memory objects with localization calculations
  // so that they are drawn in the World window
  cache_.localization_mem->state[0] = ball.loc.x;
  cache_.localization_mem->state[1] = ball.loc.y;
  cache_.localization_mem->state[2] = ball.absVel.x;
  cache_.localization_mem->state[3] = ball.absVel.y;
  cache_.localization_mem->covariance = kf_cov.cast<float>();

  //TODO: How do we handle not seeing the ball?
  //else {
    ////ball.distance = 10000.0f;
    ////ball.bearing = 0.0f;
    ////ball.loc = {10000.f, 0.f};
    //MatrixOd cov;
    //VectorOd kf_obs = kfilter_->getObs(&cov);
    //VectorSd kf_state = kfilter_->getMean();

    //auto kf_relBall = Point2D(kf_obs(STATE_X), kf_obs(STATE_Y));
    //auto globalBall = kf_relBall.relativeToGlobal(self.loc, self.orientation);
    //auto globalCov = cov; // should rotate, but now self.orientation == 0

    //ball.loc = globalBall;
    //ball.distance = kf_relBall.getMagnitude();
    //ball.bearing = kf_relBall.getDirection();
    //ball.absVel = Point2D(kf_state(STATE_VELX), kf_state(STATE_VELY));
  //}
}
