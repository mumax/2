//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package common

// Functions for manipulating vector and tensor indices in program and user space
// Author: Arne Vansteenkiste

// Indices for vector components
const (
	X = 0
	Y = 1
	Z = 2
)

// Indices for (anti-)symmetric kernel components
// when only 6 of the 9 components are stored.
const (
	XX = 0
	YY = 1
	ZZ = 2
	YZ = 3
	XZ = 4
	XY = 5
)

// Maps string to tensor index
var TensorIndex map[string]int = map[string]int{"XX": XX, "YY": YY, "ZZ": ZZ, "YZ": YZ, "XZ": XZ, "XY": XY}

// Maps sting to vector index
var VectorIndex map[string]int = map[string]int{"X": X, "Y": Y, "Z": Z}

// Maps tensor index to string
var TensorIndexStr map[int]string = map[int]string{XX: "XX", YY: "YY", ZZ: "ZZ", YZ: "YZ", XZ: "XZ", XY: "XY"}

// Maps vector index to string
var VectorIndexStr map[int]string = map[int]string{X: "X", Y: "Y", Z: "Z"}

// Swaps the X-Z values of the array.
// This transforms from user to program space and vice-versa.
func SwapXYZ(array []float64) {
	if len(array) == 3 {
		array[X], array[Z] = array[Z], array[X]
	}
	return
}

// Transforms the index between user and program space:
//	X  <-> Z
//	Y  <-> Y
//	Z  <-> X
//	XX <-> ZZ
//	YY <-> YY
//	ZZ <-> XX
//	YZ <-> XY
//	XZ <-> XZ
//	XY <-> YZ 
func SwapIndex(index int) int {
	switch index {
	default:
		panic(InputErrF("Vector/tensor index out of range:", index))
	case X:
		return Z // also handles XX
	case Y:
		return Y // also handles YY
	case Z:
		return X // also handles ZZ
	case YZ:
		return XY
	case XZ:
		return XZ
	case XY:
		return YZ
	}
	return -1 // silence 6g
}
