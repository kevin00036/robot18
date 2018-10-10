#include <localization/KalmanFilter.h>

ExtKalmanFilter::ExtKalmanFilter(TextLogger*& tlogger) : tlogger_(tlogger) {
  reset();
}

void ExtKalmanFilter::reset() {
  mu_ << 1000, 0, 0, 0;
  cov_ = MatrixSd::Zero();
  cov_(0, 0) = cov_(1, 1) = INIT_ERR_POS * INIT_ERR_POS;
  cov_(2, 2) = cov_(3, 3) = INIT_ERR_VEL * INIT_ERR_VEL;
}

VectorSd ExtKalmanFilter::transitionFunc(VectorSd mu, VectorCd ctrl, double delta_t, \
                                         MatrixSd *cov, MatrixSd *jac) {
  MatrixSd Jac, Cov;

  double adt = TRANS_ACCEL * delta_t;
  double vx = mu(STATE_VELX), vy = mu(STATE_VELY);
  double v = hypot(vx, vy) + 1E-5;
  double r = 1. - adt / v;
  double s = (r >= 0);
  double p = adt / (v * v * v);
  double px = p * vx * vx, py = p * vy * vy;
  double xrat = v * (1. - s * r * r) / (2. * TRANS_ACCEL);
  double eps = 1e-3;

  VectorSd mu_p;
  mu_p.segment<2>(0) = mu.segment<2>(0) + xrat * mu.segment<2>(2);
  mu_p.segment<2>(2) = s * r * mu.segment<2>(2);

  Jac << 1, 0, delta_t, 0,
         0, 1, 0, delta_t,
         0, 0, s * (px + r)+eps, s * px,
         0, 0, s * py, s * (py + r)+eps;

  Cov = MatrixSd::Identity() * TRANS_ERR_POS * TRANS_ERR_POS * delta_t;
  Cov(2, 2) = Cov(3, 3) = TRANS_ERR_VEL * TRANS_ERR_VEL * delta_t;

  if(cov) *cov = Cov;
  if(jac) *jac = Jac;
  return mu_p;
}

VectorOd ExtKalmanFilter::observationFunc(VectorSd mu, MatrixOd *cov, MatrixOSd *jac, bool state_noise) {
  MatrixOd Cov;
  MatrixOSd Jac;

  Jac << 1, 0, 0, 0,
         0, 1, 0, 0;

  VectorOd z = Jac * mu;

  double x = z(0), y = z(1);
  double r = hypot(x, y), theta = atan2(y, x);
  double costh = cos(theta), sinth = sin(theta);

  MatrixOd A;
  A << costh, -r*sinth,
       sinth,  r*costh;

  double rerr = max(r, 50.) * MEAS_ERR_R_RATIO, therr = MEAS_ERR_THETA;
  MatrixOd RTErr;
  RTErr << rerr*rerr, 0,
           0, therr*therr;

  tlog(30, "Rerr = %d, therr = %d", rerr, therr);

  Cov = A * RTErr * A.transpose();
  tlog(30, "MeasCov: [ %f, %f ]", Cov(0, 0), Cov(0, 1));
  tlog(30, "         [ %f, %f ]", Cov(1, 0), Cov(1, 1));

  if(state_noise)
    Cov = Jac * cov_ * Jac.transpose() + Cov;

  if(cov) *cov = Cov;
  if(jac) *jac = Jac;
  return z;
}

void ExtKalmanFilter::motionUpdate(VectorCd ctrl, double delta_t) {
  MatrixSd R, G;
  mu_ = transitionFunc(mu_, ctrl, delta_t, &R, &G);
  cov_ = G * cov_ * G.transpose() + R;
}

void ExtKalmanFilter::measureUpdate(VectorOd obs) {
  MatrixOd Q;
  MatrixOSd H;
  VectorOd h = observationFunc(mu_, &Q, &H);
  tlog(30, "StateCov: [ %f, %f ]", cov_(0, 0), cov_(0, 1));
  tlog(30, "          [ %f, %f ]", cov_(1, 0), cov_(1, 1));

  MatrixSOd SH = cov_ * H.transpose();
  MatrixSOd K = SH * (H * SH + Q).inverse();
  mu_ = mu_ + K * (obs - h);
  cov_ = (MatrixSd::Identity() - K * H) * cov_;
}

double ExtKalmanFilter::getStateLogLikelihood(VectorSd state) {
  return calcGaussianLogProb<NUM_STATE>(mu_, cov_, state);
}

double ExtKalmanFilter::getObsLogLikelihood(VectorOd obs) {
  MatrixOd Q;
  VectorOd h = observationFunc(mu_, &Q, NULL, true);
  return calcGaussianLogProb<NUM_OBS>(h, Q, obs);
}

template <int D>
double ExtKalmanFilter::calcGaussianLogProb(Matrix<double, D, 1> mu, Matrix<double, D, D> cov,
                                            Matrix<double, D, 1> x) {
  const double logSqrt2Pi = 0.5 * log(2*M_PI);
  typedef Matrix<double, D, D> Mat;
  LLT<Mat> chol(cov);
  const auto &L = chol.matrixL();
  const double quadform = L.solve(x - mu).squaredNorm();

  return -0.5 * quadform - D * logSqrt2Pi - log(L.determinant());
}
