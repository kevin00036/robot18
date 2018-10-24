#include <localization/ParticleFilter.h>
#include <memory/FrameInfoBlock.h>
#include <memory/OdometryBlock.h>
#include <common/Random.h>
#include <Eigen/Core>
#include <Eigen/Dense>

float normAngle(float x) {
  while(x >= M_PIf) x -= 2 * M_PIf;
  while(x < -M_PIf) x += 2 * M_PIf;
  return x;
}

long long get_time() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return tp.tv_sec * 1000000LL + tp.tv_usec;
}

ParticleFilter::ParticleFilter(MemoryCache& cache, TextLogger*& tlogger) 
  : cache_(cache), tlogger_(tlogger), dirty_(true) {
}

void ParticleFilter::init(Point2D loc, float orientation) {
  mean_.translation = loc;
  mean_.rotation = orientation;
  for(auto& p : particles()) {
    p.x = Random::inst().sampleU(-2500.f, 2500.f);
    p.y = Random::inst().sampleU(-1250.f, 1250.f);
    p.t = Random::inst().sampleU(-(float)M_PI, (float)M_PI);
    p.w = 1.;
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

void ParticleFilter::processFrame(vector<vector<float> > beacon_data, bool stopped, bool stopped_th, bool flying) {
  // Indicate that the cached mean needs to be updated
  dirty_ = true;
  auto start_time = get_time();
  if(flying)
    init({0.f, 0.f}, 0.f);

  // Retrieve odometry update - how do we integrate this into the filter?
  const auto& disp = cache_.odometry->displacement;
  tlog(41, "Updating particles from odometry: %2.f,%2.f @ %2.2f", disp.translation.x, disp.translation.y, disp.rotation * RAD_T_DEG);

  double dx = disp.translation.x;
  double dy = disp.translation.y;
  double dth = disp.rotation;


  for(auto& p : particles()) {
    float cs = cosf(p.t), sn = sinf(p.t);
    p.x = p.x + dx * cs - dy * sn;
    p.y = p.y + dx * sn + dy * cs;
    p.t = normAngle(p.t + dth);

    if(!stopped) {
      p.x = p.x + Random::inst().sampleN(0.f, MOTION_ERR);
      p.y = p.y + Random::inst().sampleN(0.f, MOTION_ERR);
    }
    else {
      p.x = p.x + Random::inst().sampleN(0.f, MOTION_ERR*0.5f);
      p.y = p.y + Random::inst().sampleN(0.f, MOTION_ERR*0.5f);
    }

    if(!stopped_th) {
      p.t = normAngle(p.t + Random::inst().sampleN(0.f, MOTION_ERR_TH));
    }
    else {
      p.t = normAngle(p.t + Random::inst().sampleN(0.f, MOTION_ERR_TH*0.5f));
    }
  }  
  
  for(auto& beacon : beacon_data) {
     tlog(30, "%.2f, %.2f, %.2f, %.2f", beacon[0], beacon[1], beacon[2], beacon[3]);
  }

  float sumprob = 0;
  for(auto& p : particles()) {
    float totallogprob = 0;
    for(auto& beacon : beacon_data) {
      float visdist, visbear, pardist, parbear;
      visdist = beacon[0];
      visbear = beacon[1];
      pardist = hypotf(p.x - beacon[2], p.y - beacon[3]);
      parbear = atan2f(beacon[3] - p.y, beacon[2] - p.x) - p.t;
 
      float ddis = visdist - pardist;
      float dth = normAngle(visbear - parbear);
      float sigma_dist  = SENSOR_ERR * max(visdist, 500.f);
      float sigma_theta = SENSOR_ERR_TH;
      const float logSqrt2Pi = 0.5f * logf(2*M_PIf);
      float logp_dist = -logSqrt2Pi - logf(sigma_dist) -(ddis * ddis) / (2 * sigma_dist * sigma_dist);
      float logp_theta = -logSqrt2Pi - logf(sigma_theta) -(dth * dth) / (2 * sigma_theta * sigma_theta);
      float logprob = logp_dist + logp_theta;
      totallogprob += logprob;
    }
    p.w = expf(totallogprob);
    sumprob += p.w;
  }

  for (auto& p: particles())
    p.w /= sumprob;
  
  auto& P = particles();
  int M = P.size();
  double r = Random::inst().sampleU() / M;
  int i = 0;
  double c = P[0].w;

  array<Particle, PARTICLE_NUM> new_particle;

  for(int m=0; m<M; m++){
    double u = (r + m/(double)M);
    if(!stopped) u /= 0.997;
    while(u > c and i < M-1){
      i = i + 1;
      c = c + P[i].w;
    }
    if (u <= c)
      new_particle[m] = P[i];
    else
      new_particle[m] = {
        Random::inst().sampleU(-2500.f, 2500.f), 
        Random::inst().sampleU(-1250.f, 1250.f),
        Random::inst().sampleU(-M_PIf, M_PIf)};

    new_particle[m].w = 1.f;
  }
  tlog(30,"Particle Size %d",new_particle.size());
  cache_.localization_mem->particles = new_particle;

  auto end_time = get_time();
  int proc_time = (end_time - start_time) / 100;
  cout<<"PF Time = "<<proc_time/10.<<"ms ";
  cout<<(stopped ? "STOPPED " : "        ");
  cout<<(flying ? "FLYING " : "       ");
  cout<<endl;
}

const Pose2D& ParticleFilter::pose() const {
  if(dirty_) {
    // Compute the mean pose estimate
    mean_ = mean_shift();
    dirty_ = false;
  }
  return mean_;
}

Pose2D ParticleFilter::mean_shift() const {
  float mx = 0, my = 0, mt;
  int N = PARTICLE_NUM;
  for(auto &p: particles()) {
    mx += p.x;
    my += p.y;
  }
  mx /= N;
  my /= N;
  tlog(30, "Iteration %d Mean (%.5f, %.5f)", 0, mx, my);

  for(int t=0; t<20; t++) {
    float sumx = 0, sumy = 0, sumw = 0;

    for(auto &p: particles()) {
      float w = kernel(mx, my, p.x, p.y);
      sumx += w * p.x;
      sumy += w * p.y;
      sumw += w;
    }
    mx = sumx / sumw;
    my = sumy / sumw;
    tlog(30, "Iteration %d Mean (%.5f, %.5f)", t+1, mx, my);
  }

  float sumsin = 0, sumcos = 0;
  for(auto& p: particles()) {
    float w = kernel(mx, my, p.x, p.y);
    sumcos += w * cosf(p.t);
    sumsin += w * sinf(p.t);
  }
  mt = atan2f(sumsin, sumcos);
  
  return Pose2D(mt, mx, my);
}

float ParticleFilter::kernel(float x1, float y1, float x2, float y2, float sigma) const {
  float dis = (x1-x2) * (x1-x2) + (y1-y2) * (y1-y2);
  const float sqrt2Pi = sqrtf(2 * M_PIf);
  return expf(-dis / (2.f * sigma * sigma)) / (sqrt2Pi * sigma);
}
