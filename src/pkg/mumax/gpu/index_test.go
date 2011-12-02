//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// Author: Arne Vansteenkiste

import (
	"testing"
)

func TestIndex(test *testing.T) {
	nComp := 1
	N0 := 4
	N1 := 8
	N2 := 16
	a := NewArray(nComp, []int{N0, N1, N2})
	defer a.Free()

	//fmt.Println("test X")
	SetIndexX(a)
	ah := a.LocalCopy().Array
	//fmt.Println(ah)
	for c := 0; c < nComp; c++ {
		for i := 0; i < N0; i++ {
			for j := 0; j < N1; j++ {
				for k := 0; k < N2; k++ {
					if ah[c][i][j][k] != float32(i) {
						test.Error(ah[c][i][j][k], "!=", float32(i))
					}
				}
			}
		}
	}

	//fmt.Println("test Y")
	SetIndexY(a)
	ah = a.LocalCopy().Array
	//fmt.Println(ah)
	for c := 0; c < nComp; c++ {
		for i := 0; i < N0; i++ {
			for j := 0; j < N1; j++ {
				for k := 0; k < N2; k++ {
					if ah[c][i][j][k] != float32(j) {
						test.Error(ah[c][i][j][k], "!=", float32(j))
					}
				}
			}
		}
	}

	//fmt.Println("test Z")
	SetIndexZ(a)
	ah = a.LocalCopy().Array
	//fmt.Println(ah)
	for c := 0; c < nComp; c++ {
		for i := 0; i < N0; i++ {
			for j := 0; j < N1; j++ {
				for k := 0; k < N2; k++ {
					if ah[c][i][j][k] != float32(k) {
						test.Error(ah[c][i][j][k], "!=", float32(k))
					}
				}
			}
		}
	}
}
