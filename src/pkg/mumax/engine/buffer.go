//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// A host buffer for GPU quantities, used for output.

import (
	"mumax/host"
	"sync"
)

type Buffer struct {
}

func (b *Buffer) Invalidate() {
	b.upToDate = false
}



// Gets a host array for buffering the GPU array, initializing it if necessary.
func (buffer *Buffer) Update() *host.Array {
}
