CC=g++
CFLAGS=-c -Wall 
LDFLAGS=

SOURCES=OMFHeader.cpp OMFImport.cpp
TESTSOURCES=testOMFImport.cpp test.cpp

OBJECTS=$(SOURCES:.cpp=.o)
OBJECTSSHARED=$(SOURCES:.cpp=.so)
OBJECTSSTATIC=$(SOURCES:.cpp=.a)

TESTOBJECTS=$(TESTSOURCES:.cpp=.o)

TEST1=testSimple
TEST2=testImport

all: $(SOURCES) $(TESTSOURCES) $(TEST1) $(TEST2) 

shared: $(OBJECTSSHARED)

static: $(OBJECTS) $(OBJECTSSTATIC)

%.so: %.cpp
	$(CC) $(CFLAGS) -fPIC $< -o $(@:.so=.o)
	$(CC) $(LDFLAGS) -shared -Wl,-soname,lib$@.1 -o lib$@.1.0 $(<:.cpp=.o)
	ln -sf lib$@.1.0 lib$@.1
	ln -sf lib$@.1 lib$@ 

%.a: %.o
	ar -cvq lib$@ $<

tests: $(TEST1) $(TEST2)

$(TEST1): $(OBJECTS) test.o
	$(CC) $(LDFLAGS) $(OBJECTS) test.o -o $@

$(TEST2): $(OBJECTS) testOMFImport.o
	$(CC) $(LDFLAGS) $(OBJECTS) testOMFImport.o -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

# Need -fPIC if building shared libs

clean:
	rm *.so* *.o *.a testImport testSimple