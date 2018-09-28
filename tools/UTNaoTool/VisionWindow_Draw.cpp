#include <UTMainWnd.h>
#include <VisionWindow.h>
#include <yuview/YUVImage.h>
#include <tool/Util.h>
#include <vision/Classifier.h>
#include <common/ColorSpaces.h>

#define MIN_PEN_WIDTH 3
#define IS_RUNNING_CORE (core_ && core_->vision_ && ((UTMainWnd*)parent_)->runCoreRadio->isChecked())

void VisionWindow::redrawImages() {

  if(!enableDraw_) return;

  if (((UTMainWnd*)parent_)->streamRadio->isChecked()) {
    int ms = timer_.elapsed();
    if(ms < MS_BETWEEN_FRAMES)
      return;
    timer_.start();
  }
  setImageSizes();

  redrawImages(rawImageTop,    segImageTop,    objImageTop,    horizontalBlobImageTop,    verticalBlobImageTop,    transformedImageTop);
  redrawImages(rawImageBottom, segImageBottom, objImageBottom, horizontalBlobImageBottom, verticalBlobImageBottom, transformedImageBottom);

  updateBigImage();
}

void VisionWindow::updateBigImage(ImageWidget* source) {
  ImageProcessor* processor = getImageProcessor(source);
  bigImage->setImageSource(source->getImage());
}

void VisionWindow::updateBigImage() {
  switch(currentBigImageType_) {
    case RAW_IMAGE:
      if (currentBigImageCam_ == Camera::TOP)
        updateBigImage(rawImageTop);
      else
        updateBigImage(rawImageBottom);
        break;
    case SEG_IMAGE:
      if (currentBigImageCam_ == Camera::TOP)
        updateBigImage(segImageTop);
      else
        updateBigImage(segImageBottom);
        break;
    case OBJ_IMAGE:
      if (currentBigImageCam_ == Camera::TOP)
        updateBigImage(objImageTop);
      else
        updateBigImage(objImageBottom);
        break;
    case HORIZONTAL_BLOB_IMAGE:
      if (currentBigImageCam_ == Camera::TOP)
        updateBigImage(horizontalBlobImageTop);
      else
        updateBigImage(horizontalBlobImageBottom);
        break;
    case VERTICAL_BLOB_IMAGE:
      if (currentBigImageCam_ == Camera::TOP)
        updateBigImage(verticalBlobImageTop);
      else
        updateBigImage(verticalBlobImageBottom);
        break;
    case TRANSFORMED_IMAGE:
      if (currentBigImageCam_ == Camera::TOP)
        updateBigImage(transformedImageTop);
      else
        updateBigImage(transformedImageBottom);
        break;
  }

  // draw all pixels of seg image in big window
  if (currentBigImageType_ == SEG_IMAGE){
    drawSegmentedImage(bigImage);
    if (cbxOverlay->isChecked()) {
      drawBall(bigImage);
      drawBallCands(bigImage);
      drawBeacons(bigImage);
      drawGoal(bigImage);
    }
  }

  bigImage->update();

}

void VisionWindow::redrawImages(ImageWidget* rawImage, ImageWidget* segImage, ImageWidget* objImage, ImageWidget* horizontalBlobImage, ImageWidget* verticalBlobImage, ImageWidget* transformedImage) {
  drawRawImage(rawImage);
  drawSmallSegmentedImage(segImage);
  drawGraphSegmentedImage(horizontalBlobImage, false);
  drawGraphSegmentedImage(verticalBlobImage, true);

  objImage->fill(0);
  drawBall(objImage);

  if(cbxHorizon->isChecked()) {
    drawHorizonLine(rawImage);
    drawHorizonLine(segImage);
    drawHorizonLine(horizontalBlobImage);
    drawHorizonLine(verticalBlobImage);
  }

  // if overlay is on, then draw objects on the raw and seg image as well
  if (cbxOverlay->isChecked()) {
    drawBall(rawImage);
    drawBallCands(rawImage);
    drawBeacons(rawImage);
    drawGoal(rawImage);

    drawBall(segImage);
    drawBallCands(segImage);
    drawBeacons(segImage);
    drawGoal(segImage);
  }

  drawBall(verticalBlobImage);
  drawBallCands(verticalBlobImage);

  transformedImage->fill(0);

  rawImage->update();
  segImage->update();
  objImage->update();
  horizontalBlobImage->update();
  verticalBlobImage->update();
  transformedImage->update();
}

void VisionWindow::drawRawImage(ImageWidget* widget) {
  ImageProcessor* processor = getImageProcessor(widget);
  unsigned char* image = processor->getImg();
  const ImageParams& iparams = processor->getImageParams();
  const CameraMatrix& cmatrix = processor->getCameraMatrix();
  if (!processor->isImageLoaded()) {
    widget->fill(0);
    return;
  }
  auto yuv = yuview::YUVImage::CreateFromRawBuffer(image, iparams.width, iparams.height);
  auto q = util::yuvToQ(yuv);
  widget->setImageSource(&q);
}

void VisionWindow::drawSmallSegmentedImage(ImageWidget *image) {
  ImageProcessor* processor = getImageProcessor(image);
  const ImageParams& iparams = processor->getImageParams();
  unsigned char* segImg = processor->getSegImg();
  int hstep, vstep;
  processor->color_segmenter_->getStepSize(hstep, vstep);
  if (robot_vision_block_ == NULL || segImg == NULL) {
    image->fill(0);
    return;
  }

  // This will be changed on the basis of the scan line policy
  for (int y = 0; y < iparams.height; y+=vstep) {
    for (int x = 0; x < iparams.width; x+=hstep) {
      int c = segImg[iparams.width * y + x];
      for (int smallY = 0; smallY < vstep; smallY++) {
        for (int smallX = 0; smallX < hstep; smallX++) {
          image->setPixel(x + smallX, y + smallY, segRGB[c]);
        }
      }
    }
  }
}

void VisionWindow::drawSegmentedImage(ImageWidget *image) {
  ImageProcessor* processor = getImageProcessor(image);
  const ImageParams& iparams = processor->getImageParams();
  if (doingClassification_) {
    if (image_block_ == NULL) {
      image->fill(0);
      return;
    }

    // Classify the entire image from the raw image
    unsigned char *rawImg = processor->getImg();
    unsigned char* colorTable = processor->getColorTable();
    const ImageParams& iparams = processor->getImageParams();

    for (uint16_t y = 0; y < iparams.height; y++) {
      for (uint16_t x = 0; x < iparams.width; x++) {
        Color c = ColorTableMethods::xy2color(rawImg, colorTable, x, y, iparams.width);
        image->setPixel(x, y, segRGB[c]);
      }
    }
  }
  else {
    unsigned char* segImg = processor->getSegImg();
    if (robot_vision_block_ == NULL || segImg == NULL) {
      image->fill(0);
      return;
    }

    // Seg image from memory
    for (int y = 0; y < iparams.height; y++) {
      for (int x = 0; x < iparams.width; x++) {
        int c = segImg[iparams.width * y + x];
        image->setPixel(x, y, segRGB[c]);
      }
    }
  }
  if(cbxHorizon->isChecked())
    drawHorizonLine(image);
}

void VisionWindow::drawGraphSegmentedImage(ImageWidget *image, bool classify) {
  ImageProcessor* processor = getImageProcessor(image);
  const ImageParams& iparams = processor->getImageParams();

  int* GsegImg = processor->getGSegImg();
  if (robot_vision_block_ == NULL || GsegImg == NULL) {
    image->fill(0);
    return;
  }

  // Seg image from memory
  for (int y = 0; y < iparams.height; y++) {
    for (int x = 0; x < iparams.width; x++) {
      int c = GsegImg[iparams.width * y + x];
      //unsigned char r = c * 133 + 125, g = c * 75 + 33, b = c * 193 + 179;
      int cc = (c >> 24) & 0xff;
      int yy = c & 0xff, uu = ((c >> 8) & 0xff) - 128, vv = ((c >> 16) & 0xff) - 128;
      int r = yy + 1.402f * vv;
      int g = yy - (0.344f * uu + 0.714f * vv);
      int b = yy + 1.772f * uu;
      r = r>255? 255 : r<0 ? 0 : r;
      g = g>255? 255 : g<0 ? 0 : g;
      b = b>255? 255 : b<0 ? 0 : b;


      if(classify)
        image->setPixel(x, y, segRGB[cc]);
      else
        image->setPixel(x, y, qRgb(r, g, b));

    }
  }

  if(cbxHorizon->isChecked())
    drawHorizonLine(image);
}

void VisionWindow::drawBall(ImageWidget* image) {
  if(!config_.all) return;
  if(!config_.ball) return;
  QPainter painter(image->getImage());
  painter.setPen(QPen(QColor(0, 255, 127), 3));
  if(IS_RUNNING_CORE) {
    ImageProcessor* processor = getImageProcessor(image);

    BallCandidate* best = processor->getBestBallCandidate();
    if(!best) return;

    int r = best->radius;
    painter.drawEllipse(
      (int)best->centerX - r - 1,
      (int)best->centerY - r - 1, 2 * r + 2, 2 * r + 2);
  }
  else if (world_object_block_ != NULL) {
    WorldObject* ball = &world_object_block_->objects_[WO_BALL];
    if(!ball->seen) return;
    if( (ball->fromTopCamera && _widgetAssignments[image] == Camera::BOTTOM) ||
        (!ball->fromTopCamera && _widgetAssignments[image] == Camera::TOP) ) return;
    int radius = ball->radius;
    painter.drawEllipse(ball->imageCenterX - radius, ball->imageCenterY - radius, radius * 2, radius * 2);
  }
}

void VisionWindow::drawBallCands(ImageWidget* image) {
  if(!config_.all) return;
  if(!config_.ball) return;
  QPainter painter(image->getImage());
  painter.setPen(QPen(QColor(127, 0, 255), 1));

  ImageProcessor* processor = getImageProcessor(image);

  vector<BallCandidate*> cands = processor->getBallCandidates();

  for(auto b: cands) {
    int r = b->radius;
    painter.drawEllipse(
        (int)b->centerX - r - 1,
        (int)b->centerY - r - 1, 2 * r + 2, 2 * r + 2);
  }
}

void VisionWindow::drawGoal(ImageWidget* image) {
  if(!config_.all) return;
  //if(!config_.ball) return;
  QPainter painter(image->getImage());
  painter.setPen(QPen(QColor(127, 127, 255), 3));
  if (world_object_block_ != NULL) {
    ImageProcessor* processor = getImageProcessor(image);
    WorldObject* goal = &world_object_block_->objects_[WO_OWN_GOAL];
    if(!goal->seen) return;
    if( (goal->fromTopCamera && _widgetAssignments[image] == Camera::BOTTOM) ||
        (!goal->fromTopCamera && _widgetAssignments[image] == Camera::TOP) ) return;

    int width = processor->gw, height = processor->gh;
    //QPainterPath tpath;
    //tpath.addRoundedRect(QRect(goal->imageCenterX - width/2, goal->imageCenterY - height/2, width, height), 5, 5);
    painter.drawRoundedRect(QRect(goal->imageCenterX - width/2, goal->imageCenterY - height/2, width, height), 5, 5);
    //painter.drawPath(tpath, QBrush(beacon.second[0]));
    QPen textpen(segCol[c_UNDEFINED]);
    painter.setPen(textpen);
    //painter.drawText(goal->imageCenterX - width/2, goal->imageCenterY - 10, to_string(goal->visionDistance).c_str());
    painter.drawText(goal->imageCenterX - width/2, goal->imageCenterY + 10, to_string(goal->distance).c_str());

  }
}

void VisionWindow::drawHorizonLine(ImageWidget *image) {
  if(!config_.horizon) return;
  if(!config_.all) return;
  if (robot_vision_block_ && _widgetAssignments[image] == Camera::TOP) {
    HorizonLine horizon = robot_vision_block_->horizon;
    if (horizon.exists) {
      QPainter painter(image->getImage());
      QPen wpen = QPen(segCol[c_BLUE], MIN_PEN_WIDTH);
      painter.setPen(wpen);

      ImageProcessor* processor = getImageProcessor(image);
      const ImageParams& iparams = processor->getImageParams();

      int x1 = 0;
      int x2 = iparams.width - 1;
      int y1 = horizon.gradient * x1 + horizon.offset;
      int y2 = horizon.gradient * x2 + horizon.offset;
      painter.drawLine(x1, y1, x2, y2);
    }
  }
}

void VisionWindow::drawWorldObject(ImageWidget* image, QColor color, int worldObjectID) {
  if (world_object_block_ != NULL) {
    QPainter painter(image->getImage());
    QPen wpen = QPen(color, 5);   // 2
    painter.setPen(wpen);
    WorldObject* object = &world_object_block_->objects_[worldObjectID];
    if(!object->seen) return;
    if( (object->fromTopCamera && _widgetAssignments[image] == Camera::BOTTOM) ||
        (!object->fromTopCamera && _widgetAssignments[image] == Camera::TOP) ) return;
    int offset = 10;      // 5
    int x1, y1, x2, y2;

    x1 = object->imageCenterX - offset,
    y1 = object->imageCenterY - offset,
    x2 = object->imageCenterX + offset,
    y2 = object->imageCenterY + offset;

    painter.drawLine(x1, y1, x2, y2);

    x1 = object->imageCenterX - offset,
    y1 = object->imageCenterY + offset,
    x2 = object->imageCenterX + offset,
    y2 = object->imageCenterY - offset;

    painter.drawLine(x1, y1, x2, y2);
  }
}

void VisionWindow::drawBeacons(ImageWidget* image) {
  if(!config_.all) return;
  if(world_object_block_ == NULL) return;

  map<WorldObjectType,vector<QColor>> beacons = {
    { WO_BEACON_BLUE_YELLOW, { segCol[c_BLUE], segCol[c_YELLOW] } } ,
    { WO_BEACON_YELLOW_BLUE, { segCol[c_YELLOW], segCol[c_BLUE] } },
    { WO_BEACON_BLUE_PINK, { segCol[c_BLUE], segCol[c_PINK] } },
    { WO_BEACON_PINK_BLUE, { segCol[c_PINK], segCol[c_BLUE] } },
    { WO_BEACON_PINK_YELLOW, { segCol[c_PINK], segCol[c_YELLOW] } },
    { WO_BEACON_YELLOW_PINK, { segCol[c_YELLOW], segCol[c_PINK] } }
  };
  auto processor = getImageProcessor(image);
  const auto& cmatrix = processor->getCameraMatrix();
  QPainter painter(image->getImage());
  painter.setRenderHint(QPainter::Antialiasing);
  for(auto beacon : beacons) {
    auto& object = world_object_block_->objects_[beacon.first];
    if(!object.seen) continue;
    if(object.fromTopCamera && _widgetAssignments[image] == Camera::BOTTOM) continue;
    if(!object.fromTopCamera && _widgetAssignments[image] == Camera::TOP) continue;

    QColor c1, c2;
    if(object.occlude){
      c1 = beacon.second[0].dark(180);
      c2 = beacon.second[1].dark(180);
    }
    else{
      c1 = beacon.second[0];
      c2 = beacon.second[1];
    }
    QPen tpen(c1), bpen(c2), textpen(segCol[c_UNDEFINED]);


    int width = cmatrix.getCameraWidthByDistance(object.distance, 110);
    int height = cmatrix.getCameraHeightByDistance(object.distance, 100);
    int distance = object.visionDistance;
    int distance2 = object.distance;

    Position p = cmatrix.getWorldPosition(object.imageCenterX, object.imageCenterY);
    int xc = p.x, yc = p.y, zc = p.z;

    int x1 = object.imageCenterX - width / 2;

    auto ctop = cmatrix.getImageCoordinates(xc, yc, zc + 50);
    auto cbottom = cmatrix.getImageCoordinates(xc, yc, zc - 50);
    
    // Draw top
    int ty1 = object.imageCenterY - height;
    QPainterPath tpath;
    tpath.addRoundedRect(QRect(ctop.x - width/2, object.imageCenterY - height, width, height), 5, 5);
    painter.setPen(tpen);
    painter.fillPath(tpath, QBrush(c1));

    // Draw bottom
    int by1 = object.imageCenterY, by2 = object.imageCenterY + height;
    QPainterPath bpath;
    bpath.addRoundedRect(QRect(cbottom.x - width/2, object.imageCenterY, width, height), 5, 5);
    painter.setPen(bpen);
    painter.fillPath(bpath, QBrush(c2));

    painter.setPen(textpen);
    //painter.drawText(object.imageCenterX - width/2, object.imageCenterY - 10, to_string(distance).c_str());
    painter.drawText(object.imageCenterX - width/2, object.imageCenterY + 10, to_string(distance2).c_str());
  }
  
  for(int i=-5; i<=5; i++) {
    int x = 1000, y = i * 100;
    auto ca = cmatrix.getImageCoordinates(x, y, -1000);
    auto cb = cmatrix.getImageCoordinates(x, y, 1000);
    painter.drawLine(ca.x, ca.y, cb.x, cb.y);
    //cout<<ca.x<<","<<ca.y<<" "<<cb.x<<","<<cb.y<<endl;
  }
}
