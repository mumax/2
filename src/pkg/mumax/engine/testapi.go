//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine


// Dummy API functions for testing

import (
	"mumax/host"
)


// For testing purposes.
func (api API) GetArray() *host.Array {
	return host.NewArray(3, []int{4, 5, 6})
}


// For testing purposes.
func (api API) SinkArray(a *host.Array) {
}


// For testing purposes.
func (api API) Version() int {
	return 2
}

// For testing purposes.
func (api API) Echo(i int) int {
	return i
}

// For testing purposes.
func (api API) Sink(b bool, i int, f float32, d float64, s string) {
	return
}

// For testing purposes.
func (api API) GetFloat() float32 {
	return 42.
}

// For testing purposes.
func (api API) GetDouble() float64 {
	return 42.
}

// For testing purposes.
func (api API) GetString() string {
	return "hello"
}


// For testing purposes.
func (api API) GetBool() bool {
	return true
}


// For testing purposes.
func (api API) Sum(i, j int) int {
	return i + j
}

// For testing purposes.
func (api API) GetMany() (b bool, i int, f float32, d float64, s string) {
	return true, 42, 3.12, 3.14156754326545, "hello multiple return values"
}
