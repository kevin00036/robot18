#include <localization/ParticleFilter.h>
#include <memory/FrameInfoBlock.h>
#include <memory/OdometryBlock.h>
#include <common/Random.h>
#include <Eigen/Core>
#include <Eigen/Dense>

ParticleFilter::ParticleFilter(MemoryCache& cache, TextLogger*& tlogger) 
  : cache_(cache), tlogger_(tlogger), dirty_(true) {
}

void ParticleFilter::init(Point2D loc, float orientation) {
  mean_.translation = loc;
  mean_.rotation = orientation;
  particles().resize(100);
  for(auto& p : particles()) {
    p.x = Random::inst().sampleU(-1750, 1750); //static_cast<int>(frame * 5), 250);
    p.y = Random::inst().sampleU(-1250, 1250); // 0., 250);
    p.t = Random::inst().sampleU() *2*M_PI;  //0., M_PI / 4);
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
  
  // Generate random particles for demonstration
  for(auto& beacon : beacon_data) {
     tlog(30, "%.2f, %.2f, %.2f, %.2f", beacon[0], beacon[1], beacon[2], beacon[3]);
  }

  vector<double> particleprob;
  for(auto& p : particles()) {
    double totalprob = 0;
    for(auto& beacon : beacon_data) {
      double visdist, visbear, pardist, parbear;
      visdist = beacon[0];
      visbear = beacon[1];
      pardist = sqrt((p.x - beacon[2])*(p.x - beacon[2]) + (p.y - beacon[3])*(p.y - beacon[3]));
      parbear = atan2( (beacon[3] - p.y), (beacon[2] - p.x) ) - p.t;
 

      VectorObs mu_, st_;
      MatrixObs cov_;
      mu_ << visdist, visbear;
      st_ << pardist, parbear;
      cov_ = MatrixObs::Zero();
      cov_(0, 0) = 100 * 100;
      cov_(1, 1) = M_PI/10 * M_PI/10;
      double prob = calcGaussianLogProb<2>(mu_, cov_, st_);
      totalprob += prob;
    }
    totalprob = exp(totalprob);
    particleprob.push_back(totalprob);
  }

  double sumprob = 0;
  for (auto& n : particleprob)
    sumprob += n;
  for (auto& n : particleprob)
    n = n/sumprob;
  /*
  int c = 0;
  for(auto& p : particles()) {
    p.w = particleprob[c];
    c = c+1;
  }
  */
  
  vector<Particle> new_particle;

  double M = particleprob.size();
  double r = Random::inst().sampleU()/M;
  auto& P = particles();
  int i = 0;
  double c = particleprob[0];

  for(int m=0; m<M; m++){
    double u = r + m/M;
    while(u > c){
      i = i + 1;
      c = c + particleprob[i];
    }
    new_particle.push_back(P[i]);
  }
  tlog(30,"%d",new_particle.size());
  cache_.localization_mem->particles = new_particle;


  double vx = disp.translation.x;
  double vy = disp.translation.y;
  double vth = disp.rotation;

  for(auto& p : particles()) {
    p.x = p.x + vx * cos(p.t) - vy * sin(p.t) + Random::inst().sampleN()*100;
    p.y = p.y + vx * sin(p.t) + vy * cos(p.t) + Random::inst().sampleN()*100;
    p.t = p.t + vth + Random::inst().sampleN()*M_PI/10;
  }  

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
  tlog(30, "%.2f", mean_.rotation);
  return mean_;
}


