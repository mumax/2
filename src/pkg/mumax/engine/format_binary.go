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
	"io"
)

func init() {
	RegisterOutputFormat(&FormatBinary{})
}

// Binary output format.
// The format consists of 32-bit words encoded in the machine's endianess.
// * The first word is T_MAGIC, identifying the format.
// * The second word is the rank of the encoded tensor,
// this is 4 for all mumax2 output.
// * Then as many integers as rank give the data size in each dimension.
// E.g.: 3 1 32 128 for a 3-vector field of size 128x64x1
// * Then comes the data as 32-bit floating point.
//
// The format is identical to that used in libtensor.
type FormatBinary struct{}

func (f *FormatBinary) Name() string {
	return "bin"
}

func (f *FormatBinary) Write(out io.Writer, q *Quant, options []string) {
	if len(options) > 0 {
		panic(InputErr("binary output format does not take options"))
	}
	q.Buffer().WriteBinary(out)
}
