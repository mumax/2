#include <math.h>
#include <GL/glew.h>
#include <GL/glut.h>

#include <QtGui>
#include <QtOpenGL>
#include "glwidget.h"
#include <iostream>

#ifndef GL_MULTISAMPLE
#define GL_MULTISAMPLE  0x809D
#endif

GLWidget::GLWidget(QWidget *parent)
  : QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
{
  xRot = 0;
  yRot = 0;
  zRot = 0;

  qtGreen  = QColor::fromCmykF(0.40, 0.0, 1.0, 0.0);
  qtPurple = QColor::fromCmykF(0.39, 0.39, 0.0, 0.0);
}

GLWidget::~GLWidget()
{
}

QSize GLWidget::minimumSizeHint() const
{
  return QSize(50, 50);
}

QSize GLWidget::sizeHint() const
{
  return QSize(400, 400);
}

static void qNormalizeAngle(int &angle)
{
  while (angle < 0)
    angle += 360 * 16;
  while (angle > 360 * 16)
    angle -= 360 * 16;
}

void GLWidget::setXRotation(int angle)
{
  qNormalizeAngle(angle);
  if (angle != xRot) {
    xRot = angle;
    emit xRotationChanged(angle);
    updateGL();
  }
}

void GLWidget::setYRotation(int angle)
{
  qNormalizeAngle(angle);
  if (angle != yRot) {
    yRot = angle;
    emit yRotationChanged(angle);
    updateGL();
  }
}

void GLWidget::setZRotation(int angle)
{
  qNormalizeAngle(angle);
  if (angle != zRot) {
    zRot = angle;
    emit zRotationChanged(angle);
    updateGL();
  }
}

void GLWidget::setXSliceLow(int low)
{
  if (xSliceLow != low) {
    xSliceLow = low;
    updateGL();
  }
}

void GLWidget::setXSliceHigh(int high)
{
  if (xSliceLow != high) {
    xSliceHigh = high;
    updateGL();
  }
}

void GLWidget::initializeGL()
{
  // GLUT wants argc and argv... qt obscures these in the class
  // so let us circumenvent this problem...
  int argc = 1;
  const char* argv[] = {"Sloppy","glut"};
  glutInit(&argc, (char**)argv);

  qglClearColor(qtPurple.dark());

  glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
  glColorMaterial ( GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
  glEnable(GL_COLOR_MATERIAL);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  glShadeModel(GL_SMOOTH);
  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);
  glEnable(GL_MULTISAMPLE);
  static GLfloat lightPosition[4] = { 0.5, 5.0, 7.0, 1.0 };
  glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);

  // Display List for cone
  cone = glGenLists(1);
  // Draw a cone pointing along the z axis
  glNewList(cone, GL_COMPILE);
    glPushMatrix();
    //glRotatef(0.0f,0.0f,0.0f,0.0f);
    glutSolidCone(0.2f, 0.7f, 10, 1);
    glPopMatrix();
  glEndList();

  // Fill the list of locations and spins
  numSpins=1000;
  int sx = 100;
  int sy = 5;
  int sz = 2;

  for(int i=0; i<sx; i++)
    {
      for(int j=0; j<sy; j++)
	{
	  for(int k=0; k<sz; k++)
	    {
	      int ind = k + j*sz + i*sy*sz;
	      locations[ind][0] = (float)i;
	      locations[ind][1] = (float)j;
	      locations[ind][2] = (float)k;
	      //std::cout << i << "\t" << j << "\t" << k << "\t" << ind << std::endl;
	    }
	}
    }
  
  // Find the center of mass of the object
  for(int i=0; i<numSpins; i++)
    {
      xcom += locations[i][0];
      ycom += locations[i][1];
      zcom += locations[i][2];
    }
  xcom = xcom/numSpins;
  ycom = ycom/numSpins;
  zcom = zcom/numSpins;

  // Set the slice initial conditions
  xSliceLow=0;
  xSliceHigh=16*100;
}

void GLWidget::paintGL()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glLoadIdentity();
  glTranslatef(0.0, 0.0, -10.0);
  glRotatef(xRot / 16.0, 1.0, 0.0, 0.0);
  glRotatef(yRot / 16.0, 0.0, 1.0, 0.0);
  glRotatef(zRot / 16.0, 0.0, 0.0, 1.0);
  //GLfloat testColor[] = {0.0f, 0.8f, 0.6f, 1.0f};
  //glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, testColor);
  
  // Loop over numSpins and draw each
  for (int i=0; i<numSpins; i++) {
    // Check the xSlice conditions
    if (locations[i][0] > (float)xSliceLow/16.0 && \
	locations[i][0] < (float)xSliceHigh/16.0)
      {
	glPushMatrix();
	glTranslatef(locations[i][0]-xcom, locations[i][1]-ycom, locations[i][2]-zcom);
	glColor3f(sin(locations[i][0]),cos(locations[i][0]), cos(locations[i][0]+1.0f));
	glCallList(cone);
	glPopMatrix();
      }
  }
}

void GLWidget::resizeGL(int width, int height)
{
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(60.0, (float)width / (float)height, 0.1, 80.0);
  glMatrixMode(GL_MODELVIEW);
  glViewport(0, 0, width, height);
}

void GLWidget::mousePressEvent(QMouseEvent *event)
{
  lastPos = event->pos();
}

void GLWidget::mouseMoveEvent(QMouseEvent *event)
{
  int dx = event->x() - lastPos.x();
  int dy = event->y() - lastPos.y();

  if (event->buttons() & Qt::LeftButton) {
    setXRotation(xRot + 8 * dy);
    setYRotation(yRot + 8 * dx);
  } else if (event->buttons() & Qt::RightButton) {
    setXRotation(xRot + 8 * dy);
    setZRotation(zRot + 8 * dx);
  }
  lastPos = event->pos();
}
