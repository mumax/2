//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine


import (
	"mumax/host"
	"sync"
)

var (
	bufferMutex sync.Mutex
	bufferPool  []*host.Array
)

// Fetches an array from a pool of recycled arrays.
// Should be Freed when no longer needed.
func NewBuffer(nComp int, size3D []int) *host.Array {
	bufferMutex.Lock()
	defer bufferMutex.Unlock()

	return host.NewArray(nComp, size3D)
	// todo: implement add to pool
}


// Adds an array to a pool so it can be re-used later.
func FreeBuffer(*host.Array) {
	bufferMutex.Lock()
	defer bufferMutex.Unlock()

	// todo: implement add to pool
}
