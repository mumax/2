// This part was originally created and released into the public
// domain by Gunnar Selke <gselke@physnet.uni-hamburg.de>.

#ifndef OMF_HEADER_H
#define OMF_HEADER_H

#include <string>
#include <vector>

enum OMFFormat 
{
	OMF_FORMAT_ASCII,
	OMF_FORMAT_BINARY_4,
	OMF_FORMAT_BINARY_8
};

struct OMFHeader 
{
	OMFHeader();
	~OMFHeader();

	std::string Title;
	std::vector<std::string> Desc;
	std::string meshunit; // e.g. "m"
	std::string valueunit; // e.g. "A/m"
	double valuemultiplier;
	double xmin, ymin, zmin;
	double xmax, ymax, zmax;
	double ValueRangeMaxMag, ValueRangeMinMag;
	std::string meshtype; // "rectangular"
	double xbase, ybase, zbase;
	double xstepsize, ystepsize, zstepsize;
	int xnodes, ynodes, znodes;
};

#endif
