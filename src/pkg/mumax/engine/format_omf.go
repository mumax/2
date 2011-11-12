//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// Author: Arne Vansteenkiste

import (
	. "mumax/common"
	"mumax/host"
	"unsafe"
	"io"
	"fmt"
	"strings"
)

func init() {
	RegisterOutputFormat(&FormatOmf{})
}

// OMF 1.0 Ascii output format
type FormatOmf struct{}

func (f *FormatOmf) Name() string {
	return "omf"
}

func (f *FormatOmf) Write(out io.Writer, q *Quant, options []string) {
	dataformat := ""
	switch len(options) {
	case 0: // No options: default=Binary 4
		dataformat = "Binary 4"
	case 1:
		dataformat = options[0]
	default:
		panic(InputErr(fmt.Sprint("Illegal OMF options:", options)))
	}

	writeOmfHeader(out, q)
	writeOmfData(out, q, dataformat)
	hdr(out, "End", "Segment")
}

const (
	OMF_CONTROL_NUMBER = 1234567.0 // The omf format requires the first encoded number in the binary data section to be this control number
)

func writeOmfData(out io.Writer, q *Quant, dataformat string) {

	hdr(out, "Begin", "Data "+dataformat)
	switch strings.ToLower(dataformat) {
	case "text":
		q.Buffer().WriteAscii(out)
	case "binary 4":
		writeOmfBinary4(out, q.Buffer())
	default:
		panic(InputErr("Illegal OMF data format " + dataformat + ". Options are: Text, Binary 4"))
	}
	hdr(out, "End", "Data "+dataformat)
}

// Writes the OMF header
func writeOmfHeader(out io.Writer, q *Quant) {
	gridsize := GetEngine().GridSize()

	hdr(out, "OOMMF", "rectangular mesh v1.0")
	hdr(out, "Segment count", "1")
	hdr(out, "Begin", "Segment")

	hdr(out, "Begin", "Header")

	dsc(out, "Time", GetEngine().time.Scalar())
	hdr(out, "Title", q.Name())
	hdr(out, "meshtype", "rectangular")
	hdr(out, "meshunit", "m")
	hdr(out, "xbase", 0)
	hdr(out, "ybase", 0)
	hdr(out, "zbase", 0)
	cellsize := engine.CellSize()
	hdr(out, "xstepsize", cellsize[Z])
	hdr(out, "ystepsize", cellsize[Y])
	hdr(out, "zstepsize", cellsize[X])
	hdr(out, "xmin", 0)
	hdr(out, "ymin", 0)
	hdr(out, "zmin", 0)
	hdr(out, "xmax", cellsize[Z]*float64(gridsize[Z]))
	hdr(out, "ymax", cellsize[Y]*float64(gridsize[Y]))
	hdr(out, "zmax", cellsize[X]*float64(gridsize[X]))
	hdr(out, "xnodes", gridsize[Z])
	hdr(out, "ynodes", gridsize[Y])
	hdr(out, "znodes", gridsize[X])
	hdr(out, "ValueRangeMinMag", 1e-08) // not so "optional" as the OOMMF manual suggests...
	hdr(out, "ValueRangeMaxMag", 1)     // TODO
	hdr(out, "valueunit", q.Unit())
	hdr(out, "valuemultiplier", 1)

	hdr(out, "End", "Header")
}

func writeOmfBinary4(out io.Writer, array *host.Array) {
	data := array.Array
	gridsize := array.Size3D

	var bytes []byte

	// OOMMF requires this number to be first to check the format
	var controlnumber float32 = OMF_CONTROL_NUMBER
	// Conversion form float32 [4]byte in big-endian
	// encoding/binary is too slow
	// Inlined for performance, terabytes of data will pass here...
	bytes = (*[4]byte)(unsafe.Pointer(&controlnumber))[:]
	bytes[0], bytes[1], bytes[2], bytes[3] = bytes[3], bytes[2], bytes[1], bytes[0] // swap endianess
	out.Write(bytes)

	// Here we loop over X,Y,Z, not Z,Y,X, because
	// internal in C-order == external in Fortran-order
	for i := 0; i < gridsize[X]; i++ {
		for j := 0; j < gridsize[Y]; j++ {
			for k := 0; k < gridsize[Z]; k++ {
				for c := Z; c >= X; c-- {
					// dirty conversion from float32 to [4]byte
					bytes = (*[4]byte)(unsafe.Pointer(&data[c][i][j][k]))[:]
					bytes[0], bytes[1], bytes[2], bytes[3] = bytes[3], bytes[2], bytes[1], bytes[0]
					out.Write(bytes)
				}
			}
		}
	}
}

func floats2bytes(floats []float32) []byte {
	return (*[4]byte)(unsafe.Pointer(&floats[0]))[:]
}

// Writes a header key/value pair to out:
// # Key: Value
func hdr(out io.Writer, key string, value ...interface{}) {
	fmt.Fprint(out, "# ", key, ": ")
	fmt.Fprintln(out, value...)
}

func dsc(out io.Writer, k, v interface{}) {
	hdr(out, "Desc", k, ": ", v)
}
