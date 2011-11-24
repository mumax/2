//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

// OVF2 suport added by Mykola Dvornik for mumax1, 
// modified for mumax2 by Arne Vansteenkiste, 2011.

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
	RegisterOutputFormat(&FormatOvf2{})
}

// OVF 2.0 output format
type FormatOvf2 struct{}

func (f *FormatOvf2) Name() string {
	return "ovf"
}

func (f *FormatOvf2) Write(out io.Writer, q *Quant, options []string) {
	dataformat := ""
	switch len(options) {
	case 0: // No options: default=Binary 4
		dataformat = "Binary 4"
	case 1:
		dataformat = options[0]
	default:
		panic(InputErr(fmt.Sprint("Illegal OVF options:", options)))
	}

	writeOvf2Header(out, q)
	writeOvf2Data(out, q, dataformat)
	hdr(out, "End", "Segment")
}

func writeOvf2Data(out io.Writer, q *Quant, dataformat string) {

	hdr(out, "Begin", "Data "+dataformat)
	switch strings.ToLower(dataformat) {
	case "text":
		q.Buffer().WriteAscii(out)
		//writeOmfText(out, q.Buffer())
	case "binary 4":
		writeOvf2Binary4(out, q.Buffer())
	default:
		panic(InputErr("Illegal OVF data format " + dataformat + ". Options are: Text, Binary 4"))
	}
	hdr(out, "End", "Data "+dataformat)
}

func writeOvf2Header(out io.Writer, q *Quant) {
	gridsize := GetEngine().GridSize()
	cellsize := engine.CellSize()

	fmt.Fprintln(out, "# OOMMF OVF 2.0")
	fmt.Fprintln(out, "#")
	hdr(out, "Segment count", "1")
	fmt.Fprintln(out, "#")
	hdr(out, "Begin", "Segment")
	hdr(out, "Begin", "Header")
	fmt.Fprintln(out, "#")

	hdr(out, "Title", "mumax data") // TODO
	hdr(out, "meshtype", "rectangular")
	hdr(out, "meshunit", "m")

	hdr(out, "xmin", 0)
	hdr(out, "ymin", 0)
	hdr(out, "zmin", 0)

	hdr(out, "xmax", cellsize[Z]*float64(gridsize[Z]))
	hdr(out, "ymax", cellsize[Y]*float64(gridsize[Y]))
	hdr(out, "zmax", cellsize[X]*float64(gridsize[X]))

	name := q.Name()
	var labels []interface{}
	if q.NComp() == 1 {
		labels = []interface{}{name}
	} else {
		for i := 0; i < q.NComp(); i++ {
			labels = append(labels, name+"_"+string('x'+i))
		}
	}
	hdr(out, "valuedim", q.NComp())
	hdr(out, "valuelabels", labels...) // TODO
	unit := q.Unit()
	if unit == "" {
		unit = "1"
	}
	if q.NComp() == 1 {
		hdr(out, "valueunits", unit)
	} else {
		hdr(out, "valueunits", unit, unit, unit)
	}

	totaltime := GetEngine().time.Scalar()
	// We don't really have stages
	fmt.Fprintln(out, "# Desc: Stage simulation time: ", totaltime, " s")
	fmt.Fprintln(out, "# Desc: Total simulation time: ", totaltime, " s")

	hdr(out, "xbase", cellsize[Z]/2)
	hdr(out, "ybase", cellsize[Y]/2)
	hdr(out, "zbase", cellsize[X]/2)

	hdr(out, "xnodes", gridsize[Z])
	hdr(out, "ynodes", gridsize[Y])
	hdr(out, "znodes", gridsize[X])

	hdr(out, "xstepsize", cellsize[Z])
	hdr(out, "ystepsize", cellsize[Y])
	hdr(out, "zstepsize", cellsize[X])
	fmt.Fprintln(out, "#")
	hdr(out, "End", "Header")
	fmt.Fprintln(out, "#")
}

func writeOvf2Binary4(out io.Writer, array *host.Array) {
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
	ncomp := array.NComp()
	for i := 0; i < gridsize[X]; i++ {
		for j := 0; j < gridsize[Y]; j++ {
			for k := 0; k < gridsize[Z]; k++ {
				for c := 0; c < ncomp; c++ {
					bytes = (*[4]byte)(unsafe.Pointer(&data[SwapIndex(c)][i][j][k]))[:]
					out.Write(bytes)
				}
			}
		}
	}
}
