####### Library info

LIBCPP = \
  modlm-train.cc \
  dist-train.cc \
  dist-ngram.cc \
  counts.cc \
  whitener.cc \
  dict-utils.cc \
  dist-factory.cc \
  builder-factory.cc \
  dist-one-hot.cc \
  dist-uniform.cc \
  dist-unk.cc \
  ngram-indexer.cc \
  heuristic.cc \
  timer.cc \
  macros.cc


AM_CXXFLAGS = $(BOOST_CPPFLAGS) $(EIGEN_CPPFLAGS) $(CNN_CPPFLAGS) $(OPENMP_CXXFLAGS) -I$(srcdir)/..

lib_LTLIBRARIES = libmodlm.la

libmodlm_la_SOURCES = ${LIBCPP}
libmodlm_la_LDFLAGS = -version-info 0:0:0

####### Binary info

LDADD=libmodlm.la \
    $(CNN_LDFLAGS) \
    $(BOOST_LDFLAGS) \
    $(BOOST_SYSTEM_LIB) \
    $(BOOST_PROGRAM_OPTIONS_LIB) \
    $(BOOST_SERIALIZATION_LIB) \
    $(OPENMP_CXXFLAGS)

bin_PROGRAMS = modlm-train dist-train

modlm_train_SOURCES = modlm-train-main.cc
modlm_train_LDADD = $(LDADD)

dist_train_SOURCES = dist-train-main.cc
dist_train_LDADD = $(LDADD)
