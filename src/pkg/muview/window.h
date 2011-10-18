#ifndef WINDOW_H
#define WINDOW_H

#include <QWidget>

class QSlider;
class GLWidget;
class QxtSpanSlider;
class QGroupBox;

class Window : public QWidget
{
  Q_OBJECT

  public:
  Window();

protected:
  void keyPressEvent(QKeyEvent *event);

private:
  QSlider *createSlider();
  QxtSpanSlider *createSpanSlider();

  QGroupBox *sliceGroupBox;
  QGroupBox *rotGroupBox;

  GLWidget *glWidget;
  QSlider *xSlider;
  QSlider *ySlider;
  QSlider *zSlider;
  QxtSpanSlider *xSpanSlider;
  QxtSpanSlider *ySpanSlider;
  QxtSpanSlider *zSpanSlider;

};

#endif
