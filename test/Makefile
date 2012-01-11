
CC = g++
##CPPFLAGS = -I.. -Wall -g
CPPFLAGS = -I.. -Wall -O2
ARMADILLO_LIBS = -llapack -larmadillo
##ARMADILLO_LIBS = -lgoto2 -lpthread -lgfortran
LIBS = $(ARMADILLO_LIBS)

all: linear.model.test varying.coefs.test varying.coefs.global.prior price herd herd.fast radon1

clean:
	rm -f linear.model.test linear.model.test.static varying.coefs.test price herd herd.fast radon1 varying.coefs.global.prior


logistic.model.test: logistic.model.test.cpp
	$(CC) $(CPPFLAGS) $(LIBS) logistic.model.test.cpp -o logistic.model.test

linear.model.test: linear.model.test.cpp
	$(CC) $(CPPFLAGS) $(LIBS) linear.model.test.cpp -o linear.model.test

varying.coefs.test: varying.coefs.test.cpp
	$(CC) $(CPPFLAGS) $(LIBS) varying.coefs.test.cpp -o varying.coefs.test

varying.coefs.global.prior: varying.coefs.global.prior.cpp
	$(CC) $(CPPFLAGS) $(LIBS) varying.coefs.global.prior.cpp -o varying.coefs.global.prior

price: price.cpp
	$(CC) $(CPPFLAGS) $(LIBS) price.cpp -o price

herd: herd.cpp
	$(CC) $(CPPFLAGS) $(LIBS) herd.cpp -o herd

herd.fast: herd.fast.cpp
	$(CC) $(CPPFLAGS) $(LIBS) herd.fast.cpp -o herd.fast

radon1: radon1.cpp
	$(CC) $(CPPFLAGS) $(LIBS) radon1.cpp -o radon1
