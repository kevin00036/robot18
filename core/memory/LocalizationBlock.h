#pragma once

#include <Eigen/Core>
#include <memory/MemoryBlock.h>
#include <math/Geometry.h>
#include <localization/Particle.h>
#include <schema/gen/LocalizationBlock_generated.h>
#define STATE_SIZE 4
#define COV_SIZE (STATE_SIZE * STATE_SIZE)
#define PARTICLE_NUM 1000
#define PARTICLE_DATA_SIZE (PARTICLE_NUM * 4)
#define MAX_MODELS_IN_MEM 1
#define MODEL_STATE_SIZE (MAX_MODELS_IN_MEM * STATE_SIZE)
#define MODEL_COV_SIZE (MAX_MODELS_IN_MEM * STATE_SIZE * STATE_SIZE)

class Particle;

DECLARE_INTERNAL_SCHEMA(struct LocalizationBlock : public MemoryBlock {
  public:
    SCHEMA_METHODS(LocalizationBlock);
    LocalizationBlock();
    SCHEMA_FIELD(Point2D player_);
    //SCHEMA_FIELD(Eigen::Matrix<float, STATE_SIZE, 1, Eigen::DontAlign> state);
    //SCHEMA_FIELD(Eigen::Matrix<float, STATE_SIZE, STATE_SIZE, Eigen::DontAlign> covariance);
    mutable SCHEMA_FIELD(std::array<float, STATE_SIZE> state_data);
    Eigen::Matrix<float, STATE_SIZE, 1, Eigen::DontAlign> state;

    mutable SCHEMA_FIELD(std::array<float, COV_SIZE> covariance_data);
    Eigen::Matrix<float, STATE_SIZE, STATE_SIZE, Eigen::DontAlign> covariance;

    mutable SCHEMA_FIELD(std::array<float, PARTICLE_DATA_SIZE> particles_data);
    std::array<Particle, PARTICLE_NUM> particles;

  SCHEMA_PRE_SERIALIZATION({
      std::copy(
        __source_object__.state.data(), 
        __source_object__.state.data() + __source_object__.state.size(), 
        __source_object__.state_data.data()
      );
      std::copy(
        __source_object__.covariance.data(), 
        __source_object__.covariance.data() + __source_object__.covariance.size(), 
        __source_object__.covariance_data.data()
      );
      memcpy(
        __source_object__.particles_data.data(),
        __source_object__.particles.data(), 
        __source_object__.particles_data.size() * sizeof(float)
      );
  });
  SCHEMA_POST_DESERIALIZATION({
      std::copy(
        __target_object__.state_data.data(), 
        __target_object__.state_data.data() + __target_object__.state.size(),
        __target_object__.state.data()
      );
      std::copy(
        __target_object__.covariance_data.data(), 
        __target_object__.covariance_data.data() + __target_object__.covariance.size(), 
        __target_object__.covariance.data()
      );
      memcpy(
        __target_object__.particles.data(), 
        __target_object__.particles_data.data(),
        __target_object__.particles_data.size() * sizeof(float)
      );
  });


    Point2D getBallPosition();
    Point2D getBallVel();
    Eigen::Matrix2f getBallCov();
    Eigen::Matrix2f getBallVelCov();
    //SCHEMA_FIELD(std::vector<Particle> particles);
    //void serialize(StreamBuffer& buffer, std::string);
    //bool deserialize(const StreamBuffer& buffer, std::string);
});
