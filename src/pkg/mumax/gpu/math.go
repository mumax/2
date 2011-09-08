//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// This file implements convenience wrappers for multi-GPU math functions.
// Author: Arne Vansteenkiste

import (
	"sync"
)

var add Closure
var initAdd sync.Once

// Adds 2 multi-GPU arrays: dst = a + b
func ClAdd(dst, a, b *Array) {
	initAdd.Do(func() { add = Global("math", "gpuAdd") })
	add.SetArgs(dst, a, b)
	add.Configure1D(dst.Len())
	add.Call()
}
