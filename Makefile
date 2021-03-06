# This is a make file for OnlineForest package
# Amir Saffari, amir@ymer.org
#-----------------------------------------

# Compiler options
CC = clang++

INCLUDEPATH = -I/usr/local/include
LINKPATH = -L/usr/local/lib

# PROFILING
#CFLAGS = -c -pg -O0 -Wall
#LDFLAGS = -lconfig++ -pg -latlas -llapack -lgp

# DEBUG
#CFLAGS = -c -ggdb -O0 -Wall
#LDFLAGS = -lconfig++ -latlas -llapack -lgp

# OPTIMIZED
CFLAGS = -c -O3 -Wall -march=native -mtune=native -DNDEBUG
LDFLAGS = -lconfig++ -latlas -llapack -lgp

# Source directory and files
SOURCEDIR = src
HEADERS := $(wildcard $(SOURCEDIR)/*.h)
SOURCES := $(wildcard $(SOURCEDIR)/*.cpp)
OBJECTS := $(SOURCES:.cpp=.o)

# Target output
BUILDTARGET = Online-Forest

# Build
all: $(BUILDTARGET)
$(BUILDTARGET): $(OBJECTS) $(SOURCES) $(HEADERS)
	$(CC) $(LINKPATH) $(LDFLAGS) $(OBJECTS) -o $@
.cpp.o:
	$(CC) $(CFLAGS) $(INCLUDEPATH) $< -o $@

debug:
	$(CC) -ggdb -L/usr/local/lib -lconfig++ -lf77blas -latlas -llapack -lgp src/classifier.o src/data.cpp src/hyperparameters.cpp src/Online-Forest.cpp src/onlinenode.cpp src/onlinerf.o src/onlinetree.o src/randomtest.o src/utilities.o src/mgpc.cpp src/gpc.o -o Online-Forest

clean:
	rm -f $(SOURCEDIR)/*~ $(SOURCEDIR)/*.o
	rm -f $(BUILDTARGET)
