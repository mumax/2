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

// Initiate to reduce N elements
func (r *Reductor) Init(nComp int, size []int) {
	N := (nComp * Prod(size)) / NDevice() // N floats per device
	Assert(N > 1)

	r.threads = maxThreadsPerBlock / 2 // does not work up to maxthreads

	for N <= r.threads {
		r.threads /= 2
	}
	r.blocks = DivUp(N, r.threads*2)
	r.N = N

	bufsize := []int{1, NDevice(), r.blocks}
	r.devbuffer.Init(1, bufsize, true) // true=do alloc
	r.hostbuffer.Init(1, bufsize)
}

func NewReductor(nComp int, size []int) *Reductor {
	r := new(Reductor)
	r.Init(nComp, size)
	return r
}

func (r *Reductor) Free() {
	(&(r.devbuffer)).Free()
	r.blocks = 0
	r.threads = 0
	r.N = 0
}

func (r *Reductor) Sum(in *Array) float32 {
	PartialSum(in, &(r.devbuffer), r.blocks, r.threads, r.N)
	(&r.devbuffer).CopyToHost(&r.hostbuffer)
	var sum float32
	for _, num := range r.hostbuffer.List {
		sum += num
	}
	return sum
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
