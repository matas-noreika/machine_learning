#CPP Compiler
CXX= g++
#CPP Compiler Flags
CXXFLAGS=-I./include
#Linker Dependecy Flags
CXXLDFLAGS=-L./obj -lPerceptron

#Local Directories
LOCALSRC=./src
LOCALOBJ=./obj
LOCALBIN=./bin
LOCALTESTS=./tests

#Source and Target Files
SRCS=$(wildcard $(LOCALSRC)/*.cpp)
OBJS=$(SRCS:$(LOCALSRC)/%.cpp=$(LOCALOBJ)/%.o)
LIBS=$(SRCS:$(LOCALSRC)/%.cpp=$(LOCALOBJ)/lib%.a)
TESTS:=$(wildcard $(LOCALTESTS)/*.cpp)

.PHONY: all clean build

all:
	@echo 'List of available commands:'
	@echo 'build -> generates all static libraries and test executables'
	@echo 'clean -> removes all generated files'

buildTests: $(LIBS)
	$(MAKE) -C$(LOCALTESTS)

build: clean buildTests

clean:
	-@rm -f $(LOCALOBJ)/*.o 2> /dev/null
	-@rm -f $(LOCALOBJ)/*.a 2> /dev/null
	-@rm -f $(LOCALBIN)/*.exe 2> /dev/null

$(LOCALOBJ)/%.o: $(LOCALSRC)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(LOCALOBJ)/lib%.a: $(LOCALOBJ)/%.o
	ar rcs $@ $<
