#include <localization/LocalizationModule.h>
#include <memory/WorldObjectBlock.h>
#include <memory/LocalizationBlock.h>
#include <memory/GameStateBlock.h>
#include <memory/RobotStateBlock.h>
#include <memory/RobotStateBlock.h>
#include <localization/ParticleFilter.h>
#include <localization/Logging.h>

#include <memory/BodyModelBlock.h>
#include <memory/WalkRequestBlock.h>

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

  requiresMemoryBlock("body_model");
  requiresMemoryBlock("walk_request");
}

// Boilerplate
void LocalizationModule::specifyMemoryBlocks() {
  getOrAddMemoryBlock(cache_.world_object,"world_objects");
  getOrAddMemoryBlock(cache_.localization_mem,"localization");
  getOrAddMemoryBlock(cache_.frame_info,"vision_frame_info");
  getOrAddMemoryBlock(cache_.robot_state,"robot_state");
  getOrAddMemoryBlock(cache_.game_state,"game_state");
  getOrAddMemoryBlock(cache_.odometry,"vision_odometry");

  getOrAddMemoryBlock(cache_.body_model,"body_model");
  getOrAddMemoryBlock(cache_.walk_request,"walk_request");
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
  cache_.localization_mem->player_ = Point2D(0,0);
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



// Decide whether segmant ab and segment cd intersects
bool cross(double a[],double b[], double c[], double d[]){
  double p1, p2, p3, q1, q2, q3;
  p1 = b[1]-a[1];
  p2 = a[0]-b[0];
  p3 = b[0]*a[1]-a[0]*b[1];
  q1 = d[1]-c[1];
  q2 = c[0]-d[0];
  q3 = d[0]*c[1]-c[0]*d[1];
  double sign1, sign2;
  sign1 = (p1*c[0]+p2*c[1]+p3)*(p1*d[0]+p2*d[1]+p3);
  sign2 = (q1*a[0]+q2*a[1]+q3)*(q1*b[0]+q2*b[1]+q3);
  bool cross = sign1<0 and sign2<0;
  return cross;
}



void LocalizationModule::processFrame() {
  auto& ball = cache_.world_object->objects_[WO_BALL];
  auto& self = cache_.world_object->objects_[cache_.robot_state->WO_SELF];

  // Process the current frame and retrieve our location/orientation estimate
  // from the particle filter
  bool flying = not cache_.body_model->feet_on_ground_;
  bool flying_inst = not cache_.body_model->feet_on_ground_inst_;
  tlog(30, "flying: %d %d", flying, flying_inst);

  Pose2D speed = cache_.walk_request->speed_;
  tlog(30, "speed: %.2d %.2d %.2d", speed.translation[0], speed.translation[1], speed.rotation);
  
  self.vx = speed.translation[0];
  self.vy = speed.translation[1];
  self.vth = speed.rotation;
  bool stopped = (self.vx == 0 and self.vy == 0);
  bool stopped_th = (self.vth == 0);

  self.flying = flying;
  self.flying_inst = flying_inst;

  vector<vector<float> > beacon_data;
  static map<WorldObjectType,vector<float>> beacons = {
    {WO_BEACON_BLUE_YELLOW, {2000, 1250}},
    {WO_BEACON_YELLOW_BLUE, {2000, 1250}},
    {WO_BEACON_BLUE_PINK,   {1500,-1250}},
    {WO_BEACON_PINK_BLUE,   {1500,-1250}},
    {WO_BEACON_PINK_YELLOW, {1000, 1250}},
    {WO_BEACON_YELLOW_PINK, {1000, 1250}}
  };
  for(auto beacon : beacons) {
    auto& object = cache_.world_object->objects_[beacon.first];
    if(object.seen)
      beacon_data.push_back( {object.visionDistance, object.visionBearing, beacon.second[0], beacon.second[1]} );
  }

  pfilter_->processFrame(beacon_data, stopped, stopped_th, flying);

  self.loc = pfilter_->pose().translation;
  self.orientation = pfilter_->pose().rotation;
  tlog(40, "Localization Update: x=%.2f, y=%.2f, theta=%.2f", self.loc.x, self.loc.y, self.orientation * RAD_T_DEG);


  // Calculate the time delta from last frame to this frame
  //double delta_t = (clock() - last_frame_time) / (double)CLOCKS_PER_SEC;
  //last_frame_time = clock();








  // kalman filter
  double delta_t = (clock() - last_frame_time) / (double)CLOCKS_PER_SEC;
  last_frame_time = clock();
  kfilter_->motionUpdate({}, delta_t);

  // Maintain ball seen counter (seen: +1, unseen: -1). We do the goal entering
  // detection only if the counter is >= 5
  if(ball.seen) ball_seen_counter++;
  else ball_seen_counter--;
  ball_seen_counter = max(ball_seen_counter, 0);
  ball_seen_counter = min(ball_seen_counter, 10);

  if(ball.seen) {
    last_ball_seen = clock();

    // Compute the relative position of the ball from vision readings
    auto relBall = Point2D::getPointFromPolar(ball.distance, ball.visionBearing);
    relBall = relBall.relativeToGlobal(self.loc, self.orientation);
    tlog(30, "RelBall: (%f, %f) VDis %.0f Dir %.0f", relBall.x, relBall.y, ball.visionDistance, ball.visionBearing * RAD_T_DEG);

    VectorOd obs;
    obs << relBall.x, relBall.y;
    
    kfilter_->measureUpdate(obs);
  }
  // If ball is not seen for 2 seconds, reset the kalman filter
  if(clock() - last_ball_seen >= 2.0 * CLOCKS_PER_SEC)
    kfilter_->reset();

  MatrixOd cov;
  VectorOd kf_obs = kfilter_->getObs(&cov);
  VectorSd kf_state = kfilter_->getMean();
  MatrixSd kf_cov = kfilter_->getCov();

  auto kf_relBall = Point2D(kf_obs(STATE_X), kf_obs(STATE_Y));

  // Compute the global position of the ball based on our assumed position and orientation
  auto globalBall = kf_relBall;//.relativeToGlobal(self.loc, self.orientation);
  auto globalCov = cov; // should rotate, but now self.orientation == 0

  // Update the ball in the WorldObject block so that it can be accessed in python
  ball.loc = globalBall;
  ball.distance = kf_relBall.getMagnitude();
  ball.bearing = kf_relBall.getDirection();
  ball.absVel = Point2D(kf_state(STATE_VELX), kf_state(STATE_VELY));
  ball.sd = ball.loc + (ball.absVel / TRANS_DAMP_K); // Predicted stop position


  ////////////////////////////////////////////////////////////////
  // Detect whether the ball will enter the goal (and which part)
  double a[2] = {ball.loc.x - self.loc.x, ball.loc.y - self.loc.y};
  double b[2] = {ball.sd.x - self.loc.x, ball.sd.y - self.loc.y};

  double c[2] = {0, -450};
  double d[2] = {0, 450};

  double c1[2] = {0, 0};
  double d1[2] = {0, -700};
  double c2[2] = {0, 700};
  double d2[2] = {0, 0};
  double c3[2] = {0, 180};
  double d3[2] = {0, -180};

  ball.pos = (a[0]*b[1] - b[0]*a[1])/(a[0]-b[0]);
  ball.spos = self.loc.x * ball.loc.y / ball.loc.x;

  bool right = cross(a,b,c1,d1);
  bool left = cross(a,b,c2,d2);
  bool center = cross(a,b,c3,d3);
  auto velCov = kf_cov.block<2, 2>(0, 0);

  ball.left = ball.right = ball.center = false;
  if (pow(velCov.determinant(), 1./4) <= 800 and ball_seen_counter >= 5) {
    ball.left = left;
    ball.right = right;
    ball.center = center;
  }

  tlog(30, "%d %d %d %d", ball.left, ball.right, ball.center, ball.pos);
  // Update the localization memory objects with localization calculations
  // so that they are drawn in the World window
  cache_.localization_mem->state[0] = ball.loc.x;
  cache_.localization_mem->state[1] = ball.loc.y;
  cache_.localization_mem->state[2] = ball.absVel.x;
  cache_.localization_mem->state[3] = ball.absVel.y;
  cache_.localization_mem->covariance = kf_cov.cast<float>();







}
