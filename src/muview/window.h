#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QWidget>

// Other parts of the interface
#include "preferences.h"

// For reading OMF files
#include "OMFContainer.h"
#include <vector>

// General widget stuff 
class QSlider;
class GLWidget;
class QxtSpanSlider;
class QGroupBox;

// Main Window Stuff
class QAction;
class QActionGroup;
class QLabel;
class QMenu;

class Window : public QMainWindow
{
  Q_OBJECT

  public:
  Window(int argc, char *argv[]);

protected:
  void keyPressEvent(QKeyEvent *event);
  //void contextMenuEvent(QContextMenuEvent *event);

private slots:
  void openFiles();
  void openDir();

  void settings();
  void about();
  void updateDisplayData(int index);
  
private:
  // Main Window Stuff
  void createActions();
  void createMenus();
  void adjustAnimSlider();

  QMenu *fileMenu;
  QMenu *settingsMenu;
  QMenu *helpMenu;
  
  QAction *openFilesAct;
  QAction *openDirAct;
  QAction *attachToMumax;
  QAction *settingsAct;
  QAction *aboutAct;
  //QAction *webAct;

  // Prefereences window;
  Preferences *prefs;

  // Other Stuff
  QSlider *createSlider();
  QxtSpanSlider *createSpanSlider();

  QGroupBox *sliceGroupBox;
  QGroupBox *rotGroupBox;

  GLWidget *glWidget;

  QSlider *xSlider;
  QSlider *ySlider;
  QSlider *zSlider;

  QSlider *animSlider;
  QLabel *animLabel;

  QxtSpanSlider *xSpanSlider;
  QxtSpanSlider *ySpanSlider;
  QxtSpanSlider *zSpanSlider;
  
  // Storage and caching
  std::vector<array_ptr> omfCache;
  QStringList filenames;
};

#endif
