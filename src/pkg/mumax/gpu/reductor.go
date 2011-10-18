//  This file is part of MuMax, a high-performance micromagnetic simulator
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

import (
	. "mumax/common"
)

type Reductor struct {
	operation          int
	devbuffer          Array
	hostbuffer         []float32
	blocks, threads, N int
}

// Reduces the data,
// i.e., calucates the sum, maximum, ...
// depending on the value of "operation".
func (r *Reductor) Reduce(input *Array) float32 {
	Assert(input.Len() == r.N)
	return r.reduce(r.operation, input.data, r.devbuffer, &(r.hostbuffer[0]), r.blocks, r.threads, r.N)
}

// Unsafe version of Reduce().
func (r *Reductor) reduce(data uintptr) float32 {
	return r.reduce(r.operation, data, r.devbuffer, &(r.hostbuffer[0]), r.blocks, r.threads, r.N)
}

func NewSum(b, N int) *Reductor {
	r := new(Reductor)
	r.InitSum(b, N)
	return r
}

func (r *Reductor) InitSum(b, N int) {
	r.init(b, N)
	r.operation = ADD
}

func NewSumAbs(b, N int) *Reductor {
	r := new(Reductor)
	r.InitSumAbs(b, N)
	return r
}

func (r *Reductor) InitSumAbs(b, N int) {
	r.init(b, N)
	r.operation = SUMABS
}

func NewMax(b, N int) *Reductor {
	r := new(Reductor)
	r.InitMax(b, N)
	return r
}

func (r *Reductor) InitMax(b, N int) {
	r.init(b, N)
	r.operation = MAX
}

func NewMin(b, N int) *Reductor {
	r := new(Reductor)
	r.InitMin(b, N)
	return r
}

func (r *Reductor) InitMin(b, N int) {
	r.init(b, N)
	r.operation = MIN
}

func NewMaxAbs(b, N int) *Reductor {
	r := new(Reductor)
	r.InitMaxAbs(b, N)
	return r
}

func (r *Reductor) InitMaxAbs(b, N int) {
	r.init(b, N)
	r.operation = MAXABS
}

// initiates the common pieces of all reductors
func (r *Reductor) init(b, N int) {
	Assert(N > 1)
	Assert(b != nil)

	r.threads = b.maxthreads() / 2 // does not work up to maxthreads
	if r.threads == 0 {            // for cpu and 1 thread, this becomes 0
		r.threads = 1
	}

	for N <= r.threads {
		r.threads /= 2
	}
	r.blocks = divUp(N, r.threads*2)
	r.N = N

	r.devbuffer = b.newArray(r.blocks)
	r.hostbuffer = make([]float32, r.blocks)
}
