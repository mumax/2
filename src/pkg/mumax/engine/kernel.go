//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// Author: Arne Vansteenkiste

// TODO
//  * we only need to calculate the upper triangular part
//  * we need to use the symmetry to make the calculation 8x faster
//  * return only 1/8 of the total kernel and have a function to obtain the rest via mirroring (good for storing them)

// A kernel is a rank 5 Tensor: K[S][D][x][y][z].
// S and D are source and destination directions, ranging from 0 (X) to 2 (Z).
// K[S][D][x][y][z] is the D-the component of the magnetic field at position
// (x,y,z) due to a unit spin along direction S, at the origin.
//
// As the kernel is symmetric Ksd == Kds, we only work with the upper-triangular part
//
// The kernel is usually twice the size of the magnetization field we want to convolve it with.
// The indices are wrapped: a negative index i is stored at N-abs(i), with N
// the total size in that direction.
//
// Idea: we might calculate in the kernel in double precession and only round it
// just before it is returned, or even after doing the FFT. Because it is used over
// and over, this small gain in accuracy *might* be worth it.


import ()

// Modulo-like function:
// Wraps an index to [0, max] by adding/subtracting a multiple of max.
func wrap(number, max int) int {
	for number < 0 {
		number += max
	}
	for number >= max {
		number -= max
	}
	return number
}

// Add padding x 2 in all directions where periodic == 0, except when a dimension == 1 (no padding necessary)
func padSize(size []int, periodic []int) []int {
	paddedsize := make([]int, len(size))
	for i := range size {
		if size[i] > 1 && periodic[i] == 0 {
			paddedsize[i] = 2 * size[i]
		} else {
			paddedsize[i] = size[i]
		}
	}
	return paddedsize
}
