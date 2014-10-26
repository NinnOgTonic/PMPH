CXX        = nvcc
LIB        = -L$(OPENCL_LIBDIR) -lOpenCL

SOURCES_CPP =ProjectMain.cpp ProjHelperFun.cu ProjCoreOrig.cu
HELPERS     =ProhHelperFun.h ../include/Constants.h ../include/ParseInput.h ../include/ParserC.h ../include/OpenmpUtil.h
OBJECTS     =ProjectMain.o ProjHelperFun.o  ProjCoreOrig.o
EXECUTABLE  =runproject


default: cpu

%.o: %.cu
	nvcc -g -I ../include -c --compiler-options -Wall -O3 -arch=sm_35 -o $@ $<

%.o: %.cpp
	nvcc -g -I ../include -c --compiler-options -Wall -O3 -arch=sm_35 -o $@ $<

cpu: $(EXECUTABLE)
$(EXECUTABLE): $(OBJECTS)
	nvcc -g -I ../include --compiler-options -Wall -O3 -arch=sm_35 -o $@ $^

run_small: $(EXECUTABLE)
	cat ../Data/Small/input.data ../Data/Small/output.data | ./$(EXECUTABLE)

run_medium: $(EXECUTABLE)
	cat ../Data/Medium/input.data ../Data/Medium/output.data | ./$(EXECUTABLE)

run_large: $(EXECUTABLE)
	cat ../Data/Large/input.data ../Data/Large/output.data | ./$(EXECUTABLE)

clean:
	rm -f Debug.txt $(EXECUTABLE) $(OBJECTS)
