//  This file is part of MuMax, a high-performance micromagnetic simulator
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

import (
	. "mumax/common"
	"mumax/host"
)

type Reductor struct {
	devbuffer          Array
	hostbuffer         host.Array
	blocks, threads, N int
}

// initiates the common pieces of all reductors
func (r *Reductor) Init(N int) {
	Assert(N > 1)

	r.threads = maxThreadsPerBlock / 2 // does not work up to maxthreads

	for N <= r.threads {
		r.threads /= 2
	}
	r.blocks = DivUp(N, r.threads*2)
	r.N = N

	size := []int{1, 1, r.blocks}
	r.devbuffer.Init(1, size, true)
	r.hostbuffer.Init(1, size)
}

//// Reduces the data,
//// i.e., calucates the sum, maximum, ...
//// depending on the value of "operation".
//func (r *Reductor) Reduce(input *Array) float32 {
//	Assert(input.Len() == r.N)
//	return r.reduce(r.operation, input.data, r.devbuffer, &(r.hostbuffer[0]), r.blocks, r.threads, r.N)
//}
//
//// Unsafe version of Reduce().
//func (r *Reductor) reduce(data uintptr) float32 {
//	return r.reduce(r.operation, data, r.devbuffer, &(r.hostbuffer[0]), r.blocks, r.threads, r.N)
//}
