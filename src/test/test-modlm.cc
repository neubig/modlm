#define BOOST_TEST_MODULE "modlm Tests"
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <dynet/init.h>

// Set up DyNet
struct DyNetSetup {
    DyNetSetup()   { 
        int zero = 0;
        char** null = NULL;
        dynet::initialize(zero, null);
    }
    ~DyNetSetup()  { /* shutdown your allocator/check memory leaks here */ }
};

BOOST_GLOBAL_FIXTURE( DyNetSetup );
