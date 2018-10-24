#pragma once

#include <math/Pose2D.h>
#include <memory/MemoryCache.h>
#include <memory/LocalizationBlock.h>
#include <localization/Logging.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <sys/time.h>

using namespace Eigen;


const float M_PIf = M_PI;
const float SENSOR_ERR    = 0.1f;
const float SENSOR_ERR_TH = M_PIf / 10.f;
const float MOTION_ERR    = 20.f;
const float MOTION_ERR_TH = M_PIf / 100.f;


float normAngle(float x);

long long get_time();

class ParticleFilter {
  public:
    ParticleFilter(MemoryCache& cache, TextLogger*& tlogger);
    void init(Point2D loc, float orientation);
    void processFrame(vector< vector<float> >, bool, bool);
    const Pose2D& pose() const;
    inline const std::array<Particle, PARTICLE_NUM>& particles() const {
    Pose2D mean_shift() const;
      return cache_.localization_mem->particles;
    }

  protected:
    inline std::array<Particle, PARTICLE_NUM>& particles() {
      return cache_.localization_mem->particles;
    }

  private:
    MemoryCache& cache_;
    TextLogger*& tlogger_;

    mutable Pose2D mean_;
    mutable bool dirty_;
};
