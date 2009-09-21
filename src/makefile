OPENCV = /opt/opencv
CXX = g++
LINKFLAG = -ldl -lpthread -lrt -lcv
INCFLAGS = -I"$(OPENCV)/include/opencv"
CXXFLAGS = -O3 --fast-math -msse2 

all: cgsolve_test l1qc_test l1stomp_test

clean:
	rm *.o cgsolve_test l1qc_test l1stomp_test

cgsolve_test: cgsolve_test.o cvcgsolve.o
	$(CXX) -o $@ $^ $(LINKFLAG)

l1stomp_test: l1stomp_test.o cvl1stomp.o cvcgsolve.o
	$(CXX) -o $@ $^ $(LINKFLAG)

l1qc_test: l1qc_test.o cvl1qc.o cvcgsolve.o
	$(CXX) -o $@ $^ $(LINKFLAG)

.c.o:
	$(CXX) $(INCFLAGS) $< -o $@ -c $(CXXFLAGS)

.cpp.o:
	$(CXX) $(INCFLAGS) $< -o $@ -c $(CXXFLAGS)

.cc.o:
	$(CXX) $(INCFLAGS) $< -o $@ -c $(CXXFLAGS)
