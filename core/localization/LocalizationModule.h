#pragma once

#include <Module.h>
#include <memory/MemoryCache.h>
#include <localization/LocalizationParams.h>
#include <localization/KalmanFilter.h>

class ParticleFilter;
class Point2D;

class LocalizationModule : public Module {
  public:
    LocalizationModule();
    ~LocalizationModule();
    void specifyMemoryDependency();
    void specifyMemoryBlocks();
    void initSpecificModule();
    void initFromMemory();
    void initFromWorld();
    void reInit();
    void processFrame();

    void loadParams(LocalizationParams params);
    
    void moveBall(const Point2D& position);
    void movePlayer(const Point2D& position, float orientation);
  protected:
    MemoryCache cache_;
    TextLogger*& tlogger_;
    LocalizationParams params_;
    ParticleFilter* pfilter_;
    ExtKalmanFilter* kfilter_;
    clock_t last_ball_seen;
    clock_t last_frame_time;
    int ball_seen_counter;
};
