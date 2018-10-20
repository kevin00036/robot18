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
  ball_seen_counter = 0;
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

  // Process the current frame and retrieve our location/orientation estimate
  // from the particle filter

  vector<vector<float> > beacon_data;
  static map<WorldObjectType,vector<int>> beacons = {
    {WO_BEACON_BLUE_YELLOW, {1500,1000}},
    {WO_BEACON_YELLOW_BLUE, {1500,-1000}},
    {WO_BEACON_BLUE_PINK, {0,1000}},
    {WO_BEACON_PINK_BLUE, {0,-1000}},
    {WO_BEACON_PINK_YELLOW, {-1500,1000}},
    {WO_BEACON_YELLOW_PINK, {-1500,-1000}}
  };
  for(auto beacon : beacons) {
    auto& object = cache_.world_object->objects_[beacon.first];
    if(object.seen)
      beacon_data.push_back( {object.visionDistance, object.visionBearing, beacon.second[0], beacon.second[1]} );
  }

  pfilter_->processFrame(beacon_data);

  self.loc = pfilter_->pose().translation;
  self.orientation = pfilter_->pose().rotation;
  tlog(40, "Localization Update: x=%.2f, y=%.2f, theta=%.2f", self.loc.x, self.loc.y, self.orientation * RAD_T_DEG);


  // Calculate the time delta from last frame to this frame
  double delta_t = (clock() - last_frame_time) / (double)CLOCKS_PER_SEC;
  last_frame_time = clock();

}
