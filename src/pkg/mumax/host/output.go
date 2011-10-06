//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package host

// Auhtor: Arne Vansteenkiste

import(
		. "mumax/common"
		"fmt"
		"io"
)


func (tens *Array) WriteAscii(out io.Writer){
	data := tens.Array
	gridsize := tens.Size3D

	// Here we loop over X,Y,Z, not Z,Y,X, because
	// internal in C-order == external in Fortran-order
	for i := 0; i < gridsize[X]; i++ {
		for j := 0; j < gridsize[Y]; j++ {
			for k := 0; k < gridsize[Z]; k++ {
				for c := Z; c >= X; c-- {
					fmt.Fprint(out, data[c][i][j][k], " ")
				}
				fmt.Fprint(out, "\t")
			}
			fmt.Fprint(out, "\n")
		}
		fmt.Fprint(out, "\n")
	}
}

