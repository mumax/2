# General libraries and includes\
# libqxt works well when installed from source
# but is quite finnicky in, e.g., ubuntu package manager

QT      += opengl
LIBS    += -lglut -L../libomf -lomf -lGLU

INCLUDEPATH += ../libomf

# Files and Targets
HEADERS = glwidget.h window.h \
    preferences.h qxtspanslider.h qxtspanslider_p.h
SOURCES = glwidget.cpp main.cpp window.cpp \ 
    preferences.cpp qxtspanslider.cpp

FORMS += \
    preferences.ui
