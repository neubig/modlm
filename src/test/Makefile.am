AM_CXXFLAGS = $(BOOST_CPPFLAGS) $(CNN_CPPFLAGS) $(EIGEN_CPPFLAGS) $(OPENMP_CXXFLAGS) -I$(srcdir)/..

noinst_PROGRAMS = test-modlm
TESTS = test-modlm

test_modlm_SOURCES = \
    test-modlm.cc 

test_modlm_LDADD = \
	../modlm/libmodlm.la \
    $(CNN_LDFLAGS) \
	$(BOOST_LDFLAGS) \
	$(BOOST_UNIT_TEST_FRAMEWORK_LIB) \
    $(BOOST_PROGRAM_OPTIONS_LIB) \
    $(BOOST_SERIALIZATION_LIB) \
    $(OPENMP_CXXFLAGS)
