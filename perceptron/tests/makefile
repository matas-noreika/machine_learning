#CPP Compiler
CXX= g++
#CPP Compiler Flags
CXXFLAGS=-I../include
#Linker Dependecy Flags
CXXLDFLAGS=-L$(LOCALOBJ) -lPerceptron

#Local Directories
LOCALSRC=../src
LOCALOBJ=../obj
LOCALBIN=../bin

#Source and Target Files
SRCS=$(wildcard *.cpp)
EXES=$(SRCS:%.cpp=$(LOCALBIN)/%.exe)
.PHONY: all clean

all: $(EXES)

$(LOCALBIN)/%.exe: %.cpp
	$(CXX) $(CXXFLAGS) $< $(CXXLDFLAGS) -o $@
