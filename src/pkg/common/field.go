//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package common

// This file implements "Fields". Fields are physical quantities represented by
// either scalar, vector or tensor fields in time and space.

import ()

// Conceptually, each field is represented by A(r) * m(t), a pointwise multiplication
// of an N-vector function of space A(r) by an N-vector function of time m(t).
// A(r) is an array, m(t) is the multiplier.
//
// When the array is nil/NULL, the field is independent of space. The array is then
// interpreted as 1(r), the unit field. In this way, quantities that are constant
// over space (homogeneous) can be represented.
//
// When the array has only one component and the multiplier has N components,
// then the field has N components: a(r) * m0(t), a(r) * m1(t), ... a(r) * mN(t)
//
type Field struct {
	Array
	_multiplier [FIELD_MAX_COMP]float32
	multiplier  []float32
	name        string
}


// Maximum number of components of a Field.
// 1 = scalar, 3 = vector, 6 = symmetric tensor, 9 = general tensor.
const FIELD_MAX_COMP = 9
