#include <vector>
#include <boost/multi_array.hpp>
#include <boost/smart_ptr.hpp>

typedef boost::multi_array<float, 4> array_type; // 4 dimensions
typedef array_type::index index;
typedef boost::shared_ptr<array_type> array_ptr;

