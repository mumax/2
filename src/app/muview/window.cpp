#include <QtGui>
#include <QxtSpanSlider>
#include <QKeySequence>
#include "glwidget.h"
#include "window.h"
#include <iostream>

Window::Window()
{
  glWidget = new GLWidget;

  xSlider = createSlider();
  ySlider = createSlider();
  zSlider = createSlider();
  xSpanSlider = createSpanSlider();
  ySpanSlider = createSpanSlider();
  zSpanSlider = createSpanSlider();

  connect(xSlider, SIGNAL(valueChanged(int)), glWidget, SLOT(setXRotation(int)));
  connect(glWidget, SIGNAL(xRotationChanged(int)), xSlider, SLOT(setValue(int)));
  connect(ySlider, SIGNAL(valueChanged(int)), glWidget, SLOT(setYRotation(int)));
  connect(glWidget, SIGNAL(yRotationChanged(int)), ySlider, SLOT(setValue(int)));
  connect(zSlider, SIGNAL(valueChanged(int)), glWidget, SLOT(setZRotation(int)));
  connect(glWidget, SIGNAL(zRotationChanged(int)), zSlider, SLOT(setValue(int)));

  connect(xSpanSlider, SIGNAL(lowerValueChanged(int)), glWidget, SLOT(setXSliceLow(int)));
  connect(xSpanSlider, SIGNAL(upperValueChanged(int)), glWidget, SLOT(setXSliceHigh(int)));
  connect(ySpanSlider, SIGNAL(lowerValueChanged(int)), glWidget, SLOT(setYSliceLow(int)));
  connect(ySpanSlider, SIGNAL(upperValueChanged(int)), glWidget, SLOT(setYSliceHigh(int)));
  connect(zSpanSlider, SIGNAL(lowerValueChanged(int)), glWidget, SLOT(setZSliceLow(int)));
  connect(zSpanSlider, SIGNAL(upperValueChanged(int)), glWidget, SLOT(setZSliceHigh(int)));
 
  QHBoxLayout *mainLayout = new QHBoxLayout;

  sliceGroupBox = new QGroupBox(tr("XYZ Slicing"));
  rotGroupBox   = new QGroupBox(tr("Rotation"));
  sliceGroupBox->setAlignment(Qt::AlignHCenter);
  rotGroupBox->setAlignment(Qt::AlignHCenter);

  QHBoxLayout *sliceLayout = new QHBoxLayout;
  QHBoxLayout *rotLayout   = new QHBoxLayout;

  sliceLayout->addWidget(xSpanSlider);
  sliceLayout->addWidget(ySpanSlider);
  sliceLayout->addWidget(zSpanSlider);

  rotLayout->addWidget(xSlider);
  rotLayout->addWidget(ySlider);
  rotLayout->addWidget(zSlider);

  sliceGroupBox->setLayout(sliceLayout);
  rotGroupBox->setLayout(rotLayout);
  sliceGroupBox->setFixedWidth(120);
  rotGroupBox->setFixedWidth(120);

  mainLayout->addWidget(rotGroupBox);
  mainLayout->addWidget(glWidget);
  mainLayout->addWidget(sliceGroupBox);
  setLayout(mainLayout);

  xSlider->setValue(15 * 16);
  ySlider->setValue(345 * 16);
  zSlider->setValue(0 * 16);
  setWindowTitle(tr("MuView: Mumax2 Viewer"));
}

QxtSpanSlider *Window::createSpanSlider()
{
  QxtSpanSlider *spanSlider = new QxtSpanSlider(Qt::Vertical);
  spanSlider->setRange(0 *16, 100 * 16);
  spanSlider->setSingleStep(16);
  spanSlider->setPageStep(15 * 16);
  spanSlider->setTickInterval(15 * 16);
  spanSlider->setTickPosition(QSlider::TicksRight);
  spanSlider->setHandleMovementMode(QxtSpanSlider::NoOverlapping);
  spanSlider->setLowerValue(0*16);
  spanSlider->setUpperValue(100*16);
  return spanSlider;
}

QSlider *Window::createSlider()
{
  QSlider *slider = new QSlider(Qt::Vertical);
  slider->setRange(0, 360 * 16);
  slider->setSingleStep(16);
  slider->setPageStep(15 * 16);
  slider->setTickInterval(15 * 16);
  slider->setTickPosition(QSlider::TicksRight);
  return slider;
}

void Window::keyPressEvent(QKeyEvent *e)
{
  if (e->modifiers() == Qt::CTRL) {
    // Close on Ctrl-Q or Ctrl-W
    if (e->key() == Qt::Key_Q || e->key() == Qt::Key_W )
      close();
  } else if (e->modifiers() == Qt::SHIFT) {
    if (e->key() == Qt::Key_Q || e->key() == Qt::Key_W )
      close();
  } else {
    if (e->key() == Qt::Key_Escape)
      close();
    else
      QWidget::keyPressEvent(e);
  }
}


