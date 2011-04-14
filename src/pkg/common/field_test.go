//  This file is part of MuMax, a high-perfomrance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package common

// Author: Arne Vansteenkiste

import (
	"testing"
)

func TestFieldAlloc(t *testing.T) {
	size3D := []int{4, 16, 32}
	for i := 0; i < 100; i++ {
		alpha := NewScalar("alpha")
		m := NewVectorField("m", size3D)
		h := NewField("h", 3, size3D)
		h.Init(h.Name(), 1, size3D)
		alpha.Free()
		m.Free()
		h.Free()
	}
}
