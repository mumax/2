// This part was originally created and released into the public
// domain by Gunnar Selke <gselke@physnet.uni-hamburg.de>.

#ifndef OMF_IMPORT_H
#define OMF_IMPORT_H

#include "matrix/matty.h"

#include <string>
#include <istream>

#include "OMFHeader.h"

VectorMatrix readOMF(const std::string &path, OMFHeader &header);
VectorMatrix readOMF(       std::istream &in, OMFHeader &header);

#endif

