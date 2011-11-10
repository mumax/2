# General libraries and includes\
# libqxt works well when installed from source
# but is quite finnicky in, e.g., ubuntu package manager

QT      += opengl
CONFIG  += qxt
QXT     += core gui
LIBS    += -lglut -L../../../lib -lOMFImport -lOMFHeader
INCLUDEPATH += ../../lib

# Files and Targets
HEADERS = glwidget.h window.h
SOURCES = glwidget.cpp main.cpp window.cpp 
TARGET  = muview

