CXX        = nvcc

SOURCES_CPP =ProjectMain.cpp ProjHelperFun.cu ProjCoreOrig.cu
HELPERS     =ProhHelperFun.h ../include/Constants.h ../include/ParseInput.h ../include/ParserC.h ../include/OpenmpUtil.h
OBJECTS     =ProjectMain.o ProjHelperFun.o  ProjCoreOrig.o
EXECUTABLE  =runproject
CXXFLAGS    = -g -I ../include --compiler-options -Wall -O3 -arch=sm_35

default: cpu

%.o: %.cu
	nvcc -c $(CXXFLAGS) -o $@ $<

%.o: %.cpp
	nvcc -c $(CXXFLAGS) -o $@ $<

cpu: $(EXECUTABLE)
$(EXECUTABLE): $(OBJECTS)
	nvcc $(CXXFLAGS) -o $@ $^

run_small: $(EXECUTABLE)
	cat ../Data/Small/input.data ../Data/Small/output.data | ./$(EXECUTABLE)

run_medium: $(EXECUTABLE)
	cat ../Data/Medium/input.data ../Data/Medium/output.data | ./$(EXECUTABLE)

run_large: $(EXECUTABLE)
	cat ../Data/Large/input.data ../Data/Large/output.data | ./$(EXECUTABLE)

clean:
	rm -f Debug.txt $(EXECUTABLE) $(OBJECTS)
