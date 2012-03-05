//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

import (
	. "mumax/common"
)

// Checks if size = 2^n * {1,3,5,7},
// which is very suited as CUFFT transform size.
func IsGoodCUFFTSize(n int) bool {
	if n < 1 {
		return false
	}
	for n%2 == 0 {
		n /= 2
	}
	if n%3 == 0 {
		n /= 3
	}
	if n%5 == 0 {
		n /= 5
	}
	if n%7 == 0 {
		n /= 7
	}
	return n == 1
}

// Stricter than IsGoodCUFFTSize():
// Should be a good CUFFT size and meet alignment
// requirements.
func IsGoodGridSize1(direction, n int) bool {
	if !IsGoodCUFFTSize(n) {
		return false
	}
	switch direction {
	default:
		panic(Bug("Illegal argument"))
	case Z:
		return n%16 == 0
	case Y:
		return n%16 == 0
	case X:
		return n%1 == 0
	}
	panic(Bug("Unreachable"))
	return false
}

func IsGoodGridSize(size []int) bool {
	for i := 0; i < 3; i++ {
		if !IsGoodGridSize1(i, size[i]) {
			return false
		}
	}
	return true
}
