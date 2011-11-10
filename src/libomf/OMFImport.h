#ifndef OMF_IMPORT_H
#define OMF_IMPORT_H

//#include "matrix/VectorMatrix.h"
#include "OMFContainer.h"

#include <string>
#include <istream>

#include "OMFHeader.h"

// Now we return pointers to boost arrays.
// Arrays should be garbage collected when
// we dereference the pointers everywhere

array_ptr readOMF(const std::string  &path, OMFHeader &header);
array_ptr readOMF(      std::istream   &in, OMFHeader &header);

//VectorMatrix readOMF(const std::string &path, OMFHeader &header);
//VectorMatrix readOMF(       std::istream &in, OMFHeader &header);

#endif

