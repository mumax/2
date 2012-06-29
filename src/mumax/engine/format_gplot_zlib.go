//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// Implements output for gnuplot's "splot"
// Author: Mykola Dvornik

import (
	"io"
)

func init() {
	RegisterOutputFormat(&FormatGPlotZLib{})
}

type FormatGPlotZLib struct{}

func (f *FormatGPlotZLib) Name() string {
	return "gplot.zlib"
}

func (f *FormatGPlotZLib) Write(out io.Writer, q *Quant, options []string) {
    (new (FormatGPlot)).Write(out, q, []string{"zlib"})
}
