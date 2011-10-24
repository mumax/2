// This part was originally created and released into the public
// domain by Gunnar Selke <gselke@physnet.uni-hamburg.de>.

#include "OMFImport.h"

#include "endian.h"

#include <stdlib.h>

#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <memory>
using namespace std;

struct OMFImport
{
	void read(std::istream &input);

	void parse();
	void parseSegment();
	void parseHeader();
	void parseDataAscii();
	void parseDataBinary4();
	void parseDataBinary8();

	OMFHeader header;
	std::istream *input;
	int lineno;
	std::string line;
	bool eof;
	char next_char;

	std::auto_ptr<VectorMatrix> field;

	void acceptLine();
};


VectorMatrix readOMF(const std::string &path, OMFHeader &header)
{
	std::ifstream in(path.c_str());
	if (!in.good()) {
		throw std::runtime_error(std::string("Could not open file: ") + path);
	}

	OMFImport omf;
	omf.read(in);
	header = omf.header;
	return VectorMatrix(*(omf.field));
}

VectorMatrix readOMF(std::istream &in, OMFHeader &header)
{
	OMFImport omf;
	omf.read(in);
	header = omf.header;
	return VectorMatrix(*(omf.field));
}

////////////////////////////////////////////////////////////////////////////////

static int str2int(const std::string &value)
{
	return atoi(value.c_str());
}

static double str2dbl(const std::string &value)
{
	return strtod(value.c_str(), 0);
}

static bool parseCommentLine(const std::string &line, std::string &key, std::string &value)
{
	if (line[0] == '#') {
		const int sep = line.find(':');
		key = std::string(line.begin()+2, line.begin()+sep);
		value = std::string(line.begin()+sep+2, line.end());
		return true;
	} else {
		return false;
	}
}

void OMFImport::read(std::istream &in)
{
	OMFHeader header = OMFHeader();
	input = &in;
	lineno = 0;
	eof = false;
	input->read(&next_char, sizeof(char));

	acceptLine(); // read in first line
	parse(); // Parse file
}

void OMFImport::acceptLine()
{
	static const char LF = 0x0A;
	static const char CR = 0x0D;

	// Accept LF (Unix), CR, CR+LF (Dos) and LF+CR as line terminators.
	line = "";

	bool done = false;
	while (!done) {
		if (next_char == LF) {
			done = true;
			input->read(&next_char, sizeof(char));
			if (next_char == CR) input->read(&next_char, sizeof(char));
		} else if (next_char == CR) {
			done = true;
			input->read(&next_char, sizeof(char));
			if (next_char == LF) input->read(&next_char, sizeof(char));
		} else {
			line += next_char;
			input->read(&next_char, sizeof(char));
		}
	}
/*
	input->getline(buffer, 1000);
	if (input->fail()) {
		line = "<EOF>";
		eof = true;
		return;
	}

	line = buffer;
	lineno += 1;
*/
}

// OMF file parser /////////////////////////////////////////////////////////////////////////////

void OMFImport::parse()
{
	bool ok;
	std::string key, value;
	
	ok = parseCommentLine(line, key, value);
	if (ok && key == "OOMMF") {
		acceptLine();
	} else {
		throw std::runtime_error("Expected 'OOMMF' at line 1");
	}

	ok = parseCommentLine(line, key, value);
	if (ok && key == "Segment count") {
		acceptLine();
	} else {
		throw std::runtime_error("Expected 'Segment count' at line 2");
	}

	ok = parseCommentLine(line, key, value);
	if (ok && key == "Begin" && value == "Segment") {
		parseSegment();
	} else {
		throw std::runtime_error("Expected begin of segment");
	}
}

void OMFImport::parseSegment()
{
	bool ok;
	std::string key, value;

	ok = parseCommentLine(line, key, value);
	if (!ok || key != "Begin" || value != "Segment") {
		throw std::runtime_error("Parse error. Expected 'Begin Segment'");
	}
	acceptLine();

	parseHeader();
	ok = parseCommentLine(line, key, value);
	if (!ok || key != "Begin") {
		throw std::runtime_error("Parse error. Expected 'Begin Data <type>'");
	}
	if (value == "Data Text") {
		parseDataAscii();
	} else if (value == "Data Binary 4") {
		parseDataBinary4();
	} else if (value == "Data Binary 8") {
		parseDataBinary8();
	} else {
		throw std::runtime_error("Expected either 'Text', 'Binary 4' or 'Binary 8' chunk type");
	}

	ok = parseCommentLine(line, key, value);
	if (!ok || key != "End" || value != "Segment") {
		throw std::runtime_error("Expected 'End Segment'");
	}
	acceptLine();
}

void OMFImport::parseHeader()
{
	bool ok;
	std::string key, value;

	ok = parseCommentLine(line, key, value);
	if (!ok || key != "Begin" || value != "Header") {
		throw std::runtime_error("Expected 'Begin Header'");
	}
	acceptLine();
	
	bool done = false;
	while (!done) {
		ok = parseCommentLine(line, key, value);
		if (!ok) {
			cout << "Skipped line." << endl;
			continue;
		}

		if (key == "End" && value == "Header") {
			done = true;
			break;
		} else if (key == "Title") {
			header.Title = value;
		} else if (key == "Desc") {
			header.Desc.push_back(value);
		} else if (key == "meshunit") {
			header.meshunit = value;
		} else if (key == "valueunit") {
			header.valueunit = value;
		} else if (key == "valuemultiplier") {
			header.valuemultiplier = str2dbl(value);
		} else if (key == "xmin") {
			header.xmin = str2dbl(value);
		} else if (key == "ymin") {
			header.ymin = str2dbl(value);
		} else if (key == "zmin") {
			header.zmin = str2dbl(value);
		} else if (key == "xmax") {
			header.xmax = str2dbl(value);
		} else if (key == "ymax") {
			header.ymax = str2dbl(value);
		} else if (key == "zmax") {
			header.zmax = str2dbl(value);
		} else if (key == "ValueRangeMinMag") {
			header.ValueRangeMinMag = str2dbl(value);
		} else if (key == "ValueRangeMaxMag") {
			header.ValueRangeMaxMag = str2dbl(value);
		} else if (key == "meshtype") {
			header.meshtype = value;
		} else if (key == "xbase") {
			header.xbase = str2dbl(value);
		} else if (key == "ybase") {
			header.ybase = str2dbl(value);
		} else if (key == "zbase") {
			header.zbase = str2dbl(value);
		} else if (key == "xstepsize") {
			header.xstepsize = str2dbl(value);
		} else if (key == "ystepsize") {
			header.ystepsize = str2dbl(value);
		} else if (key == "zstepsize") {
			header.zstepsize = str2dbl(value);
		} else if (key == "xnodes") {
			header.xnodes = str2int(value);
		} else if (key == "ynodes") {
			header.ynodes = str2int(value);
		} else if (key == "znodes") {
			header.znodes = str2int(value);
		} else {
			clog << "OMFImport::parseHeader: Unknown key: " << key << "/" << value << endl;
		}
		acceptLine();
	}

	ok = parseCommentLine(line, key, value);
	if (!ok || key != "End" || value != "Header") {
		throw std::runtime_error("Expected 'End Header'");
	}
	acceptLine();
}

void OMFImport::parseDataAscii()
{
	bool ok;
	std::string key, value;
	
	ok = parseCommentLine(line, key, value);
	if (!ok || key != "Begin" || value != "Data Text") {
		throw std::runtime_error("Expected 'Begin DataText'");
	}
	acceptLine();

	// Create field matrix object
	field.reset(new VectorMatrix(Shape(header.xnodes, header.ynodes, header.znodes)));
	field->clear();

	VectorMatrix::accessor field_acc(*field);

	for (int z=0; z<header.znodes; ++z)
	for (int y=0; y<header.ynodes; ++y)
	for (int x=0; x<header.xnodes; ++x) {
		std::stringstream ss;
		ss << line;

		double v1, v2, v3;
		ss >> v1 >> v2 >> v3;
		Vector3d vec(v1, v2, v3);
		
		vec = vec * header.valuemultiplier;
		field_acc.set(x, y, z, vec);

		acceptLine();
	}

	ok = parseCommentLine(line, key, value);
	if (!ok || key != "End" || value != "Data Text") {
		throw std::runtime_error("Expected 'End Data Text'");
	}
	acceptLine();
}

void OMFImport::parseDataBinary4()
{
	assert(sizeof(float) == 4);

	bool ok;
	std::string key, value;

	// Parse "Begin: Data Binary 4"
	ok = parseCommentLine(line, key, value);
	if (!ok || key != "Begin" || value != "Data Binary 4") {
		throw std::runtime_error("Expected 'Begin Binary 4'");
	}

	// Create field matrix object
	field.reset(new VectorMatrix(Shape(header.xnodes, header.ynodes, header.znodes)));
	field->clear();

	const int num_cells = field->size();

	// Read magic value and field contents from file
	float magic; 
	((char*)&magic)[0] = next_char; next_char = -1;
	input->read((char*)&magic+1, sizeof(char)); 
	input->read((char*)&magic+2, sizeof(char)); 
	input->read((char*)&magic+3, sizeof(char)); 
	magic = fromBigEndian(magic);

	if (magic != 1234567.0f) throw std::runtime_error("Wrong magic number (binary 4 format)");

	float *buffer = new float [3*num_cells];
	input->read((char*)buffer, 3*num_cells*sizeof(float));

	VectorMatrix::accessor field_acc(*field);

	for (int i=0; i<num_cells; ++i) {
		Vector3d vec;
		vec.x = fromBigEndian(buffer[i*3+0]);
		vec.y = fromBigEndian(buffer[i*3+1]);
		vec.z = fromBigEndian(buffer[i*3+2]);
		field_acc.set(i, vec * header.valuemultiplier);
	}

	delete [] buffer;

	acceptLine(); // read trailing newline character
	acceptLine(); // read next line...

	// Parse "End: Data Binary 4"
	ok = parseCommentLine(line, key, value);
	if (!ok || key != "End" || value != "Data Binary 4") {
		throw std::runtime_error("Expected 'End Data Binary 4'");
	}
	acceptLine();
}

void OMFImport::parseDataBinary8()
{
	assert(sizeof(double) == 8);

	bool ok;
	std::string key, value;

	// Parse "Begin: Data Binary 8"
	ok = parseCommentLine(line, key, value);
	if (!ok || key != "Begin" || value != "Data Binary 8") {
		throw std::runtime_error("Expected 'Begin Binary 8'");
	}

	// Create field matrix object
	field.reset(new VectorMatrix(Shape(header.xnodes, header.ynodes, header.znodes)));
	field->clear();

	const int num_cells = field->size();

	// Read magic value and field contents from file
	double magic;
	((char*)&magic)[0] = next_char; next_char = -1;
	input->read((char*)&magic+1, sizeof(char)); 
	input->read((char*)&magic+2, sizeof(char)); 
	input->read((char*)&magic+3, sizeof(char)); 
	input->read((char*)&magic+4, sizeof(char)); 
	input->read((char*)&magic+5, sizeof(char)); 
	input->read((char*)&magic+6, sizeof(char)); 
	input->read((char*)&magic+7, sizeof(char)); 
	magic = fromBigEndian(magic);

	if (magic != 123456789012345.0) throw std::runtime_error("Wrong magic number (binary 8 format)");

	double *buffer = new double [3*num_cells];
	input->read((char*)buffer, 3*num_cells*sizeof(double));

	VectorMatrix::accessor field_acc(*field);

	for (int i=0; i<num_cells; ++i) {
		Vector3d vec;
		vec.x = fromBigEndian(buffer[i*3+0]);
		vec.y = fromBigEndian(buffer[i*3+1]);
		vec.z = fromBigEndian(buffer[i*3+2]);
		field_acc.set(i, vec * header.valuemultiplier);
	}

	delete [] buffer;

	acceptLine(); // read trailing newline character
	acceptLine(); // read next line...

	// Parse "End: Data Binary 8"
	ok = parseCommentLine(line, key, value);
	if (!ok || key != "End" || value != "Data Binary 8") {
		throw std::runtime_error("Expected 'End Data Binary 8'");
	}
	acceptLine();
}

