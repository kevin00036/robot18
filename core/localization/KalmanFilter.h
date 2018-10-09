#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <localization/Logging.h>

using namespace Eigen;

enum KalmanState {
  STATE_X,
  STATE_Y,
  STATE_VELX,
  STATE_VELY,

  NUM_STATE
};

enum KalmanObs {
  OBS_X,
  OBS_Y,

  NUM_OBS
};

enum KalmanCtrl {

  NUM_CTRL
};

typedef Matrix<double, NUM_STATE, NUM_STATE> MatrixSd;
typedef Matrix<double, NUM_STATE, 1> VectorSd;
typedef Matrix<double, NUM_OBS, NUM_OBS> MatrixOd;
typedef Matrix<double, NUM_OBS, 1> VectorOd;
typedef Matrix<double, NUM_STATE, NUM_OBS> MatrixSOd;
typedef Matrix<double, NUM_OBS, NUM_STATE> MatrixOSd;
typedef Matrix<double, NUM_CTRL, 1> VectorCd;

const double INIT_ERR_POS = 10000.;
const double INIT_ERR_VEL = 1000.;
const double TRANS_DAMP_K = 1.;
const double TRANS_ERR_POS = 250.;
const double TRANS_ERR_VEL = 500.;
const double MEAS_ERR_R_RATIO = 0.1;
const double MEAS_ERR_THETA = 0.1;

class ExtKalmanFilter {
 public:
  ExtKalmanFilter(TextLogger*& tlogger);

  void reset();

  VectorSd transitionFunc(VectorSd mu, VectorCd ctrl, double delta_t, MatrixSd *cov=NULL, MatrixSd *jac=NULL);
  VectorOd observationFunc(VectorSd mu, MatrixOd *cov=NULL, MatrixOSd *jac=NULL, bool state_noise=false);

  void motionUpdate(VectorCd ctrl, double delta_t);
  void measureUpdate(VectorOd obs);

  VectorSd getMean() {return mu_;};
  MatrixSd getCov() {return cov_;};
  VectorOd getObs(MatrixOd *cov=NULL) {return observationFunc(mu_, cov, NULL, true);};
  
  double getStateLogLikelihood(VectorSd state);
  double getObsLogLikelihood(VectorOd obs);

  template <int D>
    double calcGaussianLogProb(Matrix<double, D, 1> mu, Matrix<double, D, D> cov, Matrix<double, D, 1> x);


 private:
  VectorSd mu_;
  MatrixSd cov_;
  TextLogger*& tlogger_;
  
};
