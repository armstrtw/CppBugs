
CC = g++
##CPPFLAGS = -I.. -Wall -g
CPPFLAGS = -I.. -Wall -O2
ARMADILLO_LIBS = -llapack
##ARMADILLO_LIBS = -lgoto2 -lpthread -lgfortran
LIBS = $(ARMADILLO_LIBS)

all: linear.model.test linear.model.test.static varying.coefs.test price herd herd.fast

clean:
	rm -f linear.model.test linear.model.test.static varying.coefs.test price herd herd.fast

linear.model.test: linear.model.test.cpp
	$(CC) $(CPPFLAGS) $(LIBS) linear.model.test.cpp -o linear.model.test

linear.model.test.static: linear.model.test.static.cpp
	$(CC) $(CPPFLAGS) $(LIBS) linear.model.test.static.cpp -o linear.model.test.static

varying.coefs.test: varying.coefs.test.cpp
	$(CC) $(CPPFLAGS) $(LIBS) varying.coefs.test.cpp -o varying.coefs.test

price: price.cpp
	$(CC) $(CPPFLAGS) $(LIBS) price.cpp -o price

herd: herd.cpp
	$(CC) $(CPPFLAGS) $(LIBS) herd.cpp -o herd

herd.fast: herd.fast.cpp
	$(CC) $(CPPFLAGS) $(LIBS) herd.fast.cpp -o herd.fast
