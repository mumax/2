//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2012  Arne Vansteenkiste, Ben Van de Wiele and Mykola Dvornik.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// Implements binary dump output,
// in the machine's endianess.
// uses 32-bit words:
//	0: magic "#d1\n"
//	1: rank: 4
//	2:2+rank: sizes for each direction
//	rest: ieee float data.
// Author: Arne Vansteenkiste

import (
	"io"
	. "mumax/common"
	"unsafe"
)

func init() {
	RegisterOutputFormat(&FormatDump{})
}

type FormatDump struct{}

func (f *FormatDump) Name() string {
	return "dump"
}

func (f *FormatDump) Write(out io.Writer, q *Quant, options []string) {
	if len(options) > 0 {
		panic(InputErr("dump output format does not take options"))
	}

	data := q.Buffer().Array
	list := q.Buffer().List

	out.Write([]byte("#d1\n"))
	writeInt(out, 4) // rank 4
	writeInt(out, len(data))
	writeInt(out, len(data[0]))
	writeInt(out, len(data[0][0]))
	writeInt(out, len(data[0][0][0]))
	out.Write( (*(*[1<<31-1]byte)(unsafe.Pointer(&list[0])))[0:4*len(list)] )
}

