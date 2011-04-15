//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package common

// This file implements a "Universe". 
// A universe defines time, space (size of discretization grid)
// and a number of physical fields that are defined on the grid.
// 
// E.g.: a Universe may have size 1 x 64 x 64 and contain a magnetization,
// field and energy density, each of that size.
//
// Author: Arne Vansteenkiste.

import (
	//. "mumax/common"
)

type Universe struct{
	_size3D [3]int // INTERNAL
	size3D []int // Discretization grid size

	timeId int // Integer representation of time ("number of time steps taken")
	time float64 // Time in internal units

	Units

	
}
