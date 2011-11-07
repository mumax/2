CC=g++
CFLAGS=-c -Wall 
LDFLAGS=

SOURCES=OMFHeader.cpp OMFImport.cpp
TESTSOURCES=testOMFImport.cpp test.cpp

OBJECTS=$(SOURCES:.cpp=.o)
OBJECTSSHARED=$(SOURCES:.cpp=.so)

TESTOBJECTS=$(TESTSOURCES:.cpp=.o)

TEST1=testSimple
TEST2=testImport

all: $(SOURCES) $(TESTSOURCES) $(TEST1) $(TEST2) 

shared: $(OBJECTS) $(OBJECTSSHARED)

%.so: %.o
	$(CC) $(LDFLAGS) -shared -Wl,-soname,lib$@.1 -o lib$@.1.0 $<
	ln -sf lib$@.1.0 lib$@.1
	ln -sf lib$@.1 lib$@ 

tests: $(TEST1) $(TEST2)

$(TEST1): $(OBJECTS) test.o
	$(CC) $(LDFLAGS) $(OBJECTS) test.o -o $@

$(TEST2): $(OBJECTS) testOMFImport.o
	$(CC) $(LDFLAGS) $(OBJECTS) testOMFImport.o -o $@

.cpp.o:
	$(CC) $(CFLAGS) -fPIC $< -o $@

clean:
	rm *.so* *.o testImport testSimple