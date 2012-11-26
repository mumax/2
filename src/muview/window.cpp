#include <deque>
#include <iostream>

#include <QtGui>
#include <QDir>
//#include <QxtSpanSlider>
#include <QKeySequence>
#include <QFileDialog>
#include <QFileInfo>
#include <QFileSystemWatcher>
#include <QSignalMapper>
#include "glwidget.h"
#include "window.h"
#include <QDebug>
#include <QTimer>
#include <QMap>

#include "OMFImport.h"
#include "OMFHeader.h"
#include "OMFContainer.h"
#include "qxtspanslider.h"

struct OMFImport;

Window::Window(int argc, char *argv[])
{
  QWidget *widget = new QWidget;
  setCentralWidget(widget);

  // Cache size
  cacheSize = 50;
  cachePos  = 0;

  glWidget = new GLWidget;
  prefs = new Preferences(this);

  xSlider = createSlider();
  ySlider = createSlider();
  zSlider = createSlider();
  xSpanSlider = createSpanSlider();
  ySpanSlider = createSpanSlider();
  zSpanSlider = createSpanSlider();

  // Rotation
  connect(xSlider, SIGNAL(valueChanged(int)), glWidget, SLOT(setXRotation(int)));
  connect(glWidget, SIGNAL(xRotationChanged(int)), xSlider, SLOT(setValue(int)));
  connect(ySlider, SIGNAL(valueChanged(int)), glWidget, SLOT(setYRotation(int)));
  connect(glWidget, SIGNAL(yRotationChanged(int)), ySlider, SLOT(setValue(int)));
  connect(zSlider, SIGNAL(valueChanged(int)), glWidget, SLOT(setZRotation(int)));
  connect(glWidget, SIGNAL(zRotationChanged(int)), zSlider, SLOT(setValue(int)));

  // Slicing
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

  QVBoxLayout *displayLayout = new QVBoxLayout;
  QHBoxLayout *sliceLayout   = new QHBoxLayout;
  QHBoxLayout *rotLayout     = new QHBoxLayout;

  // Center display and animation bar
  animLabel  = new QLabel(tr("<i>Animation</i> timeline"));
  animLabel->setAlignment(Qt::AlignCenter);
  animSlider = new QSlider(Qt::Horizontal);
  animSlider->setRange(0, 10);
  animSlider->setSingleStep(1);
  animSlider->setPageStep(10);
  animSlider->setTickInterval(2);
  animSlider->setTickPosition(QSlider::TicksRight);
  animSlider->setEnabled(FALSE);
  animLabel->setFixedHeight(animLabel->sizeHint().height());
  displayLayout->addWidget(glWidget);
  displayLayout->addWidget(animLabel);
  displayLayout->addWidget(animSlider);

  // Slicing
  sliceLayout->addWidget(xSpanSlider);
  sliceLayout->addWidget(ySpanSlider);
  sliceLayout->addWidget(zSpanSlider);
  sliceGroupBox->setLayout(sliceLayout);
  sliceGroupBox->setFixedWidth(120);

  // Rotation
  rotLayout->addWidget(xSlider);
  rotLayout->addWidget(ySlider);
  rotLayout->addWidget(zSlider);
  rotGroupBox->setLayout(rotLayout);
  rotGroupBox->setFixedWidth(120);

  // Overall Layout
  mainLayout->addWidget(rotGroupBox);
  mainLayout->addLayout(displayLayout);
  mainLayout->addWidget(sliceGroupBox);
  widget->setLayout(mainLayout);

  // Main Window Related
  createActions();
  createMenus();

  xSlider->setValue(15 * 16);
  ySlider->setValue(345 * 16);
  zSlider->setValue(0 * 16);
  setWindowTitle(tr("MuView: Mumax2 Viewer"));

  // Data, don't connect until we are ready (probably still not ready here)...
  connect(animSlider, SIGNAL(valueChanged(int)), this, SLOT(updateDisplayData(int)));
  
  // Load files from command line if supplied
  if (argc > 1) {
    OMFHeader tempHeader = OMFHeader();
    QStringList rawList;
    for (int i=1; i<argc; i++) {
      rawList << argv[i];
    }

    if (rawList.contains(QString("-w"))) {
        if (rawList.indexOf("-w") < (rawList.length() - 1))  {
            watchDir(rawList[rawList.indexOf("-w")+1]);
        }
    } else {
        QStringList allLoadedFiles;
        foreach (QString item, rawList)
        {
            QFileInfo info(item);
            if (!info.exists()) {
                std::cout << "File " << item.toStdString() << " does not exist" << std::endl;
            } else {
                // Push our new content...
                if (info.isDir()) {
                    QDir chosenDir(item);
                    dirString = chosenDir.path()+"/";
                    QStringList filters;
                    filters << "*.omf" << "*.ovf";
                    chosenDir.setNameFilters(filters);
                    QStringList files = chosenDir.entryList();

                    foreach (QString file, files)
                    {
                        filenames << (dirString+file);
                        displayNames << (dirString+item);
                        //omfCache.push_back(readOMF((dirString+file).toStdString(), tempHeader));
                    }

                } else {
                    // just a normal file
                    filenames << (dirString+item);
                    displayNames << (dirString+item);
                    //omfCache.push_back(readOMF(item.toStdString(), tempHeader));
                }
            }
        }
        // persistent storage of filenames for top overlay
        //filenames = allLoadedFiles;

        // Looping over files
        OMFHeader tempHeader = OMFHeader();
        for (int loadPos=0; loadPos<cacheSize && loadPos<filenames.size(); loadPos++) {
            omfCache.push_back(readOMF((filenames[loadPos]).toStdString(), tempHeader));
            //qDebug() << QString("Pushing Back") << filenames[loadPos];
        }

        // Update the Display with the first element
        glWidget->updateData(omfCache.front());
        // Update the top overlay
        glWidget->updateTopOverlay(filenames.front());
        // Refresh the animation bar
        adjustAnimSlider();
    }
  }
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

void Window::adjustAnimSlider()
{
  //int cacheSize = (int)omfCache.size();
  //std::cout << "Cache size is:\t" << cacheSize << std::endl;
  int numFiles = filenames.size();
  //qDebug() << QString("Updating Animation Slider to size") << numFiles;
  if (numFiles > 1) {
    animSlider->setRange(0, numFiles-1);
    animSlider->setSingleStep(1);
    animSlider->setPageStep(10);
    animSlider->setTickInterval(2);
    animSlider->setTickPosition(QSlider::TicksRight);
    animSlider->setEnabled(TRUE);
    animSlider->setSliderPosition(0);
  } else {
    animSlider->setEnabled(FALSE);
  }
  //qDebug() << QString("Updated Animation Slider to size") << numFiles;
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

void Window::createMenus()
{
  fileMenu = menuBar()->addMenu(tr("&File"));
  fileMenu->addAction(openFilesAct);
  fileMenu->addAction(openDirAct);
  fileMenu->addAction(watchDirAct);
  fileMenu->addSeparator();

  settingsMenu = menuBar()->addMenu(tr("&Settings"));
  
  helpMenu = menuBar()->addMenu(tr("&Help"));
  helpMenu->addAction(aboutAct);
  helpMenu->addSeparator();
  //helpMenu->addAction(webAct);

  settingsMenu->addAction(settingsAct);
  settingsMenu->addSeparator();
  settingsMenu->addAction(cubesAct);
  settingsMenu->addAction(conesAct);
}

void Window::about()
{
  //infoLabel->setText(tr("Invoked <b>Help|About</b>"));
  QMessageBox::about(this, tr("About Muview"),
		     tr("<b>Muview</b> 0.1 \n<br>"
			"Mumax visualization tool written in OpenGL and Qt<br>"
			"<br>Created by Graham Rowlands 2011."));
}

void Window::settings()
{
    prefs->exec();
}

void Window::openFiles()
{
  QString fileName;
  fileName = QFileDialog::getOpenFileName(this,
					  tr("Open File"), QDir::currentPath(),
					  tr("OVF Files (*.omf *.ovf)"));
  
  if (fileName != "") 
    {
      OMFHeader tempHeader = OMFHeader();

      // Remove the last element if not empty
      while (!omfCache.empty()) {
        omfCache.pop_back();
      }

      // Push our file data
      omfCache.push_back(readOMF(fileName.toStdString(), tempHeader));
      // Update the Display with the first element
      glWidget->updateData(omfCache.back());
      
      // Refresh the animation bar
      adjustAnimSlider();

      // Refresh the overlay
      glWidget->updateTopOverlay("");
    }
}

void Window::updateWatchedFiles(const QString& str) {
    // Look at all of the files in the directory
    // and add those which are not in the list of
    // original filenames

    // filenames contains the list of loaded files
    // watchedFiles is a map of files to load and their modification timestamps

    // When the timestamps in wathcedFiles stop changing we actually
    // push the relevant files into the OMF cache.

    QDir chosenDir(str);
    QString dirString = chosenDir.path()+"/";
    QStringList filters;
    filters << "*.omf" << "*.ovf";
    chosenDir.setNameFilters(filters);
    QStringList dirFiles = chosenDir.entryList();
    OMFHeader tempHeader = OMFHeader();

    // compare to existing list of files
    bool changed = false;
    foreach(QString dirFile, dirFiles)
    {
            //qDebug() << QString("Changed File!") << dirFile;
            if (!filenames.contains(dirFile)) {
                // Haven't been loaded
                QString fullPath = dirString + dirFile;
                QFileInfo info(fullPath);
                //qDebug() << QString("Not in filenames: ") << (dirFile);

                if (!watchedFiles.contains(fullPath)) {
                    // Not on the watch list yet
                    watchedFiles.insert(fullPath,info.lastModified());
                    //qDebug() << QString("Inserted: ") << (dirFile);
                } else {
                    // on the watch list
                    if (info.lastModified() == watchedFiles[fullPath]) {
                        // File size has stabalized
                        //qDebug() << QString("Stable, pushed") << dirFile;
                        filenames.append(dirFile);
                        omfCache.push_back(readOMF(fullPath.toStdString(), tempHeader));
                        changed = true;
                    } else {
                        // File still changing
                        //qDebug() << QString("Unstable") << dirFile;
                        watchedFiles[fullPath] = info.lastModified();
                    }
                }
            }
    }

    if (changed) {

        //qDebug() << QString("Updated Data");
        noFollowUpdate = false;
        if (!noFollowUpdate) {
            // Update the Display with the first element
            glWidget->updateData(omfCache.back());
            // Update the top overlay
            glWidget->updateTopOverlay(displayNames.back());
        }
        // Refresh the animation bar
        adjustAnimSlider();
    }

}

void Window::updateDisplayData(int index)
{
  // Check to see if we've cached this data already.
  // Add and remove elements from the front and back
  // of the deque until we've caught up... if we're
  // too far out of range just scratch everything and
  // reload.

  OMFHeader tempHeader = OMFHeader();
  if ( abs(index-cachePos) >= cacheSize ) {
      // Out of the realm of caching
      // Clear the cache of pre-existing elements
      //qDebug() << QString("Clearing the cache, too far out of range") << index << cachePos;
      while (!omfCache.empty()) {
        omfCache.pop_back();
      }
      cachePos = index;
      for (int loadPos=index; loadPos<(index+cacheSize) && loadPos<filenames.size(); loadPos++) {
          omfCache.push_back(readOMF((filenames[loadPos]).toStdString(), tempHeader));
      }
      cachePos = index;
  } else if ( index < cachePos ) {
      // Moving backwards, regroup for fast scrubbing!
      //qDebug() << QString("Moving backwards") << index << cachePos;
      for (int loadPos=cachePos-1; loadPos >= index && loadPos<filenames.size(); loadPos--) {
          if (omfCache.size()==uint(cacheSize)) {
             omfCache.pop_back();
          } else {
             //qDebug() << QString("Refilling");
          }
          omfCache.push_front(readOMF((filenames[loadPos]).toStdString(), tempHeader));
      }
      cachePos = index;
  }

  // We should be within the current cache
  if (index < filenames.size()) {
    //qDebug() << QString("In Cache Range") << index << cachePos;
    // Update the top overlay
    glWidget->updateTopOverlay(displayNames[index]);
    // Update the Display
    //qDebug() << QString("Current cache size") << omfCache.size();
    glWidget->updateData(omfCache.at(index-cachePos));
  } else {
      //qDebug() << QString("Out of Cache Range!!!!") << index << cachePos;
      glWidget->updateTopOverlay(QString("Don't scroll so erratically..."));
  }
}

void Window::openDir()
{
  QString dir = QFileDialog::getExistingDirectory(this, tr("Open Directory"),
                                                 "/home",
                                                 QFileDialog::ShowDirsOnly
                                                 | QFileDialog::DontResolveSymlinks);
  
  if (dir != "") 
    {
      QDir chosenDir(dir);
      dirString = chosenDir.path()+"/";
      QStringList filters;
      filters << "*.omf" << "*.ovf";
      chosenDir.setNameFilters(filters);
      QStringList dirFiles = chosenDir.entryList();
      filenames.clear();
      foreach (QString file, dirFiles) {
         filenames.push_back(dirString + file);
      }

      // persistent storage of filenames for top overlay
      displayNames = dirFiles;

      cachePos  = 0; // reset position to beginning

      // Clear the cache of pre-existing elements
      while (!omfCache.empty()) {
        omfCache.pop_back();
      }

      // Looping over files
      OMFHeader tempHeader = OMFHeader();
      for (int loadPos=0; loadPos<cacheSize && loadPos<filenames.size(); loadPos++) {
          omfCache.push_back(readOMF((filenames[loadPos]).toStdString(), tempHeader));
          //qDebug() << QString("Pushing Back") << filenames[loadPos];
      }
//      foreach (QString file, filenames)
//      {
//            //std::cout << (dirString+file).toStdString() << std::endl;
//            // Push our new content...
//            if (loadPos < cacheSize) {
//                omfCache.push_back(readOMF((dirString+file).toStdString(), tempHeader));
//            }
//            loadPos++;
//      }

      // Update the Display with the first element
      glWidget->updateData(omfCache.front());
  
      // Update the top overlay
      glWidget->updateTopOverlay(displayNames.front());

      // Refresh the animation bar
      //qDebug() << QString("Updating Animation Slider");
      adjustAnimSlider();
    }
}

void Window::watchDir(const QString& str)
{
    QString dir;
    // Don't show a dialog if we get this message from the command line
    if (str == "") {
     dir = QFileDialog::getExistingDirectory(this, tr("Watch Directory"),
                                                     "/home",
                                                     QFileDialog::ShowDirsOnly
                                                     | QFileDialog::DontResolveSymlinks);
    } else {
        dir = str;
    }

  if (dir != "")
    {
      // Added the dir to the watch list
      watcher = new QFileSystemWatcher();
      //waiter = new QFileSystemWatcher();
      watcher->addPath(dir);

      // Now read all of the current files
      QDir chosenDir(dir);
      QString dirString = chosenDir.path()+"/";
      QStringList filters;
      filters << "*.omf" << "*.ovf";
      chosenDir.setNameFilters(filters);
      QStringList dirFiles = chosenDir.entryList();

      // persistent storage of filenames for top overlay
      filenames = dirFiles;

      if (filenames.length()>0) {
          // Clear the cache of pre-existing elements
          while (!omfCache.empty()) {
              omfCache.pop_back();
          }

          // Qt macro for looping over files
          OMFHeader tempHeader = OMFHeader();
          foreach (QString file, dirFiles)
          {
              //std::cout << (dirString+file).toStdString() << std::endl;
              // Push our new content...
              omfCache.push_back(readOMF((dirString+file).toStdString(), tempHeader));
          }

          // Update the Display with the first element
          glWidget->updateData(omfCache.front());

          // Update the top overlay
          glWidget->updateTopOverlay(filenames.front());

          // Refresh the animation bar
          adjustAnimSlider();
      }
      // Now the callbacks
      QObject::connect(watcher, SIGNAL(directoryChanged(QString)),
              this, SLOT(updateWatchedFiles(QString)));
    }

}

void Window::toggleDisplay() {
  glWidget->toggleDisplay(cubesAct->isChecked());
}


void Window::createActions()
{
  aboutAct = new QAction(tr("&About Muview"), this);
  connect(aboutAct, SIGNAL(triggered()), this, SLOT(about()));

  settingsAct = new QAction(tr("&Muview Preferences"), this);
  connect(settingsAct, SIGNAL(triggered()), this, SLOT(settings()));

  cubesAct = new QAction(tr("&Display Cubes"), this);
  conesAct = new QAction(tr("&Display Cones"), this);
  cubesAct->setCheckable(true);
  cubesAct->setChecked(true);
  conesAct->setCheckable(true);
  connect(cubesAct, SIGNAL(triggered()), this, SLOT(toggleDisplay()));
  connect(conesAct, SIGNAL(triggered()), this, SLOT(toggleDisplay()));
  displayType = new QActionGroup(this);
  displayType->addAction(cubesAct);
  displayType->addAction(conesAct);

  openFilesAct  = new QAction(tr("&Open File(s)"), this);
  openFilesAct->setShortcuts(QKeySequence::Open);
  connect(openFilesAct, SIGNAL(triggered()), this, SLOT(openFiles()));

  openDirAct  = new QAction(tr("&Open Dir"), this);
  openDirAct->setShortcut( QKeySequence(Qt::CTRL + Qt::Key_D) );
  connect(openDirAct, SIGNAL(triggered()), this, SLOT(openDir()));

  QSignalMapper* signalMapper = new QSignalMapper (this);
  watchDirAct  = new QAction(tr("&Follow Dir"), this);
  watchDirAct->setShortcut( QKeySequence(Qt::CTRL + Qt::Key_F) );
  connect(watchDirAct, SIGNAL(triggered()), signalMapper, SLOT(map()));
  signalMapper->setMapping (watchDirAct, "") ;
  connect (signalMapper, SIGNAL(mapped(QString)), this, SLOT(watchDir(QString))) ;
}

