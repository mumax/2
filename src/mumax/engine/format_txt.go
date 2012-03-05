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
	RegisterOutputFormat(&FormatTxt{})
}

// Ascii output format
type FormatTxt struct{}

func (f *FormatTxt) Name() string {
	return "txt"
}

func (f *FormatTxt) Write(out io.Writer, q *Quant, options []string) {
	if len(options) > 0 {
		panic(InputErr("txt output format does not take options"))
	}
	q.Buffer().WriteAscii(out)
}
