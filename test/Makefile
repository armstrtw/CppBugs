
CC = g++
##CPPFLAGS = -I.. -Wall -g
CPPFLAGS = -I.. -Wall -O2 -std=c++0x
ARMADILLO_LIBS = -larmadillo
##ARMADILLO_LIBS = -lgoto2 -lpthread -lgfortran
LIBS = $(ARMADILLO_LIBS)

all: linear.model.test varying.coefs.test varying.coefs.global.prior price herd herd.fast radon1

clean:
	rm -f linear.model.test linear.model.test.static varying.coefs.test price herd herd.fast radon1 varying.coefs.global.prior


logistic.model.test: logistic.model.test.cpp
	$(CC) $(CPPFLAGS) $(LIBS) logistic.model.test.cpp -o logistic.model.test

linear.model.test: linear.model.test.cpp
	$(CC) $(CPPFLAGS) linear.model.test.cpp -o linear.model.test $(LIBS)

varying.coefs.test: varying.coefs.test.cpp
	$(CC) $(CPPFLAGS) varying.coefs.test.cpp -o varying.coefs.test $(LIBS)

varying.coefs.global.prior: varying.coefs.global.prior.cpp
	$(CC) $(CPPFLAGS) varying.coefs.global.prior.cpp -o varying.coefs.global.prior $(LIBS)

price: price.cpp
	$(CC) $(CPPFLAGS) price.cpp -o price $(LIBS)

herd: herd.cpp
	$(CC) $(CPPFLAGS) herd.cpp -o herd $(LIBS)

herd.fast: herd.fast.cpp
	$(CC) $(CPPFLAGS) herd.fast.cpp -o herd.fast $(LIBS)

radon1: radon1.cpp
	$(CC) $(CPPFLAGS) radon1.cpp -o radon1 $(LIBS)
