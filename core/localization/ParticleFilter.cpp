#include <localization/ParticleFilter.h>
#include <memory/FrameInfoBlock.h>
#include <memory/OdometryBlock.h>
#include <common/Random.h>
#include <Eigen/Core>
#include <Eigen/Dense>

double normAngle(double x) {
  while(x >= M_PI) x -= 2 * M_PI;
  while(x < -M_PI) x += 2 * M_PI;
  return x;
}

ParticleFilter::ParticleFilter(MemoryCache& cache, TextLogger*& tlogger) 
  : cache_(cache), tlogger_(tlogger), dirty_(true) {
}

void ParticleFilter::init(Point2D loc, float orientation) {
  mean_.translation = loc;
  mean_.rotation = orientation;
  particles().resize(300);
  for(auto& p : particles()) {
    p.x = Random::inst().sampleU(-1750.f, 1750.f); //static_cast<int>(frame * 5), 250);
    p.y = Random::inst().sampleU(-1250.f, 1250.f); // 0., 250);
    p.t = Random::inst().sampleU(-(float)M_PI, (float)M_PI);  //0., M_PI / 4);
    p.w = Random::inst().sampleU();
  }
}

template <int D>
double calcGaussianLogProb(Matrix<double, D, 1> mu, Matrix<double, D, D> cov,
                                            Matrix<double, D, 1> x) {
  const double logSqrt2Pi = 0.5 * log(2*M_PI);
  typedef Matrix<double, D, D> Mat;
  LLT<Mat> chol(cov);
  const auto &L = chol.matrixL();
  const double quadform = L.solve(x - mu).squaredNorm();

  return -0.5 * quadform - D * logSqrt2Pi - log(L.determinant());
}

void ParticleFilter::processFrame(vector<vector<float> > beacon_data) {
  // Indicate that the cached mean needs to be updated
  dirty_ = true;

  // Retrieve odometry update - how do we integrate this into the filter?
  const auto& disp = cache_.odometry->displacement;
  tlog(41, "Updating particles from odometry: %2.f,%2.f @ %2.2f", disp.translation.x, disp.translation.y, disp.rotation * RAD_T_DEG);

  double dx = disp.translation.x;
  double dy = disp.translation.y;
  double dth = disp.rotation;

  for(auto& p : particles()) {
    p.x = p.x + dx * cos(p.t) - dy * sin(p.t) + Random::inst().sampleN()*30;
    p.y = p.y + dx * sin(p.t) + dy * cos(p.t) + Random::inst().sampleN()*30;
    p.t = normAngle(p.t + dth + Random::inst().sampleN()*M_PI/50);
  }  
  
  for(auto& beacon : beacon_data) {
     tlog(30, "%.2f, %.2f, %.2f, %.2f", beacon[0], beacon[1], beacon[2], beacon[3]);
  }

  double sumprob = 0;
  for(auto& p : particles()) {
    double totalprob = 0;
    for(auto& beacon : beacon_data) {
      double visdist, visbear, pardist, parbear;
      visdist = beacon[0];
      visbear = beacon[1];
      pardist = hypot(p.x - beacon[2], p.y - beacon[3]);
      parbear = atan2(beacon[3] - p.y, beacon[2] - p.x) - p.t;
 
      VectorObs mu_, st_;
      MatrixObs cov_;
      double dth = normAngle(visbear - parbear);
      mu_ << visdist, 0.;
      st_ << pardist, dth;
      cov_ = MatrixObs::Zero();
      cov_(0, 0) = 100 * 100;
      cov_(1, 1) = M_PI/20 * M_PI/20;
      double prob = calcGaussianLogProb<2>(mu_, cov_, st_);
      totalprob += prob;
    }
    p.w = exp(totalprob);
    sumprob += p.w;
  }

  for (auto& p: particles())
    p.w /= sumprob;
  
  vector<Particle> new_particle;

  auto& P = particles();
  double M = P.size();
  double r = Random::inst().sampleU() / M;
  int i = 0;
  double c = P[0].w;

  for(int m=0; m<M; m++){
    double u = (r + m/M) / 0.99;
    while(u > c and i < M-1){
      i = i + 1;
      c = c + P[i].w;
    }
    if (u <= c)
      new_particle.push_back(P[i]);
    else
      new_particle.push_back({
                             Random::inst().sampleU(-2500.f, 2500.f), 
                             Random::inst().sampleU(-1250.f, 1250.f),
                             Random::inst().sampleU(-(float)M_PI, (float)M_PI)});

    new_particle.back().w = 1.;
  }
  tlog(30,"Particle Size %d",new_particle.size());
  cache_.localization_mem->particles = new_particle;
}

const Pose2D& ParticleFilter::pose() const {
  if(dirty_) {
    // Compute the mean pose estimate
    mean_ = Pose2D();
    using T = decltype(mean_.translation);
    for(const auto& p : particles()) {
      mean_.translation += T(p.x,p.y);
      mean_.rotation += p.t;
    }
    if(particles().size() > 0)
      mean_ /= static_cast<float>(particles().size());
    dirty_ = false;
  }
  return mean_;
}


