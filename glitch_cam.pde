import processing.video.*;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.CvType;
import org.opencv.core.Size;
import org.opencv.objdetect.CascadeClassifier;

import javax.imageio.ImageIO;
import java.io.*;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.awt.image.Raster;

Capture cap;
int pixCnt;
BufferedImage bm;

CascadeClassifier faceDetector;
MatOfRect faceDetections;

void setup() {
  size(640, 480);
  System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
  println(Core.VERSION);

  cap = new Capture(this, width, height);
  cap.start();
  bm = new BufferedImage(width, height, BufferedImage.TYPE_4BYTE_ABGR);
  pixCnt = width*height*4;

  faceDetector = new CascadeClassifier(dataPath("haarcascade_frontalface_default.xml"));
  faceDetections = new MatOfRect();
}

void convert(PImage _i) {
  bm.setRGB(0, 0, _i.width, _i.height, _i.pixels, 0, _i.width);
  Raster rr = bm.getRaster();
  byte [] b1 = new byte[pixCnt];
  rr.getDataElements(0, 0, _i.width, _i.height, b1);
  Mat m1 = new Mat(_i.height, _i.width, CvType.CV_8UC4);
  m1.put(0, 0, b1);

  Mat m2 = new Mat(_i.height, _i.width, CvType.CV_8UC1);
  Imgproc.cvtColor(m1, m2, Imgproc.COLOR_BGRA2GRAY);   

  faceDetector.detectMultiScale(m2, faceDetections, 3, 1, 
  Objdetect.CASCADE_DO_CANNY_PRUNING, new Size(40, 40), new Size(240, 240));

  bm.flush();
  m2.release();
  m1.release();
}

boolean facesDetected(PImage image) {
  convert(image);
  return faceDetections.toArray().length > 0;
}

PImage shiftBits(PImage img, int numberOfBytesToShift) {
  BufferedImage bimg = new BufferedImage( img.width, img.height, BufferedImage.TYPE_INT_RGB);    
  img.loadPixels();
  bimg.setRGB( 0, 0, img.width, img.height, img.pixels, 0, img.width);
  ByteArrayOutputStream baStream = new ByteArrayOutputStream();
  BufferedOutputStream bos = new BufferedOutputStream(baStream);
  try {
    ImageIO.write(bimg, "jpg", bos);

    byte[] jpegBytes = baStream.toByteArray();

    for (int i = 0; i < numberOfBytesToShift; i++) {
      int index = (int) random(jpegBytes.length - 1);
      //int index = (int) random(20);
      jpegBytes[index] = (byte) (jpegBytes[index] ^ (1 << 2));
    }

    ByteArrayInputStream inputStream = new ByteArrayInputStream(jpegBytes);
    BufferedImage shiftedBufferedImage = ImageIO.read(inputStream);
    PImage result = new PImage(shiftedBufferedImage.getWidth(), 
    shiftedBufferedImage.getHeight(), 
    PConstants.ARGB);
    shiftedBufferedImage.getRGB(0, 0, result.width, result.height, result.pixels, 0, result.width);
    result.updatePixels();
    return result;
  } 
  catch (Exception e) {
    e.printStackTrace();
    return img;
  }
}

void draw() {
  if (!cap.available()) 
    return;
  background(0);
  cap.read();

  if (facesDetected(cap)) {
    image(shiftBits(cap, 1), 0, 0);
  } else {
    image(cap, 0, 0);
  }
}

