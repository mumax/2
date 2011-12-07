#include <vector>
#include <vector>
#include <vector>
#include <iostream>

#include <QtGui>
#include <QDir>
#include <QxtSpanSlider>
#include <QKeySequence>
#include <QFileDialog>
#include <QFileInfo>
#include "glwidget.h"
#include "window.h"

#include "OMFImport.h"
#include "OMFHeader.h"
#include "OMFContainer.h"

struct OMFImport;

Window::Window(int argc, char *argv[])
{
  QWidget *widget = new QWidget;
  setCentralWidget(widget);

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
    QStringList rawList;
    for (int i=1; i<argc; i++) {
      rawList << argv[i];
    }

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
	    QString dirString = chosenDir.path()+"/";
	    QStringList filters;
	    filters << "*.omf" << "*.ovf";
	    chosenDir.setNameFilters(filters);
	    QStringList dirFiles = chosenDir.entryList();

	    OMFHeader tempHeader = OMFHeader();
	    foreach (QString file, dirFiles)
	      {  
		allLoadedFiles << file;
		omfCache.push_back(readOMF((dirString+file).toStdString(), tempHeader));
	      }

	  } else {
	    // just a normal file
	    OMFHeader tempHeader = OMFHeader();
	    allLoadedFiles << item;
	    omfCache.push_back(readOMF(item.toStdString(), tempHeader));
	  }
	}
      }
    // persistent storage of filenames for top overlay
    filenames = allLoadedFiles;
    // Update the Display with the first element
    glWidget->updateData(omfCache.front());
    // Update the top overlay
    glWidget->updateTopOverlay(filenames.front());
    // Refresh the animation bar
    adjustAnimSlider();
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
  int cacheSize = (int)omfCache.size();
  std::cout << "Cache size is:\t" << cacheSize << std::endl;

  if (cacheSize > 1) {
    animSlider->setRange(0, cacheSize-1);
    animSlider->setSingleStep(1);
    animSlider->setPageStep(10);
    animSlider->setTickInterval(2);
    animSlider->setTickPosition(QSlider::TicksRight);
    animSlider->setEnabled(TRUE);
  } else {
    animSlider->setEnabled(FALSE);
  }

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

void Window::updateDisplayData(int index)
{
  if (!omfCache.empty()) {
    int cacheSize = (int)omfCache.size();
    if (index < cacheSize) {
      // Update the top overlay
      glWidget->updateTopOverlay(filenames[index]);
      // Update the Display
      glWidget->updateData(omfCache.at(index));
    }
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
      QString dirString = chosenDir.path()+"/";
      QStringList filters;
      filters << "*.omf" << "*.ovf";
      chosenDir.setNameFilters(filters);
      QStringList dirFiles = chosenDir.entryList();

      // persistent storage of filenames for top overlay
      filenames = dirFiles;

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
  connect(openDirAct, SIGNAL(triggered()), this, SLOT(openDir()));
}

