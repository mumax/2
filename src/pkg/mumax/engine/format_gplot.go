//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// Implements output for gnuplot's "splot"
// Author: Arne Vansteenkiste

import (
	. "mumax/common"
	"io"
	"fmt"
)

func init() {
	RegisterOutputFormat(&FormatGPlot{})
}

type FormatGPlot struct{}

func (f *FormatGPlot) Name() string {
	return "gplot"
}

func (f *FormatGPlot) Write(out io.Writer, q *Quant, options []string) {
	if len(options) > 0 {
		panic(InputErr("gplot output format does not take options"))
	}

	data := q.Buffer().Array
	gridsize := GetEngine().GridSize()
	cellsize := GetEngine().CellSize()
	ncomp := len(data)

	// Here we loop over X,Y,Z, not Z,Y,X, because
	// internal in C-order == external in Fortran-order
	for i := 0; i < gridsize[X]; i++ {
		x := float64(i) *cellsize[X]
		for j := 0; j < gridsize[Y]; j++ {
			y := float64(j) *cellsize[Y]
			for k := 0; k < gridsize[Z]; k++ {
				z := float64(k) *cellsize[Z]
				_, err := fmt.Fprint(out, z, " ", y, " ", x, "\t")
				if err != nil {
					panic(IOErr(err.String()))
				}
				for c := 0; c < ncomp; c++ {
					_, err := fmt.Fprint(out, data[SwapIndex(c, ncomp)][i][j][k], " ") // converts to user space.
					if err != nil {
						panic(IOErr(err.String()))
					}
				}
				_, err = fmt.Fprint(out, "\n")
				if err != nil {
					panic(IOErr(err.String()))
				}
			}
			_, err := fmt.Fprint(out, "\n")
			if err != nil {
				panic(IOErr(err.String()))
			}
		}
		_, err := fmt.Fprint(out, "\n")
		if err != nil {
			panic(IOErr(err.String()))
		}
	}

}
