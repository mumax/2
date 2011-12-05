# General libraries and includes\
# libqxt works well when installed from source
# but is quite finnicky in, e.g., ubuntu package manager

QT      += opengl
CONFIG  += qxt
QXT     += core gui

LIBS    += -lglut -L../libomf -lomf -lQxtCore -lQxtGui

INCLUDEPATH += ../libomf /usr/include/qxt/QxtCore /usr/include/qxt/QxtGui

# Files and Targets
HEADERS = glwidget.h window.h \
    preferences.h
SOURCES = glwidget.cpp main.cpp window.cpp \ 
    preferences.cpp
TARGET  = muview

FORMS += \
    preferences.ui
