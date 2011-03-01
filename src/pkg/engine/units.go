//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// This file implements internal units.
// 
// Internal units are used to avoid that the numerical value of any
// quantity in the simulation becomes too large or too small.
// 
// In a previous implementation, the internal units were defined so that:
// Msat == Aexch == mu0 == gamma0 == 1
// However, this required Msat, Aexch to be defined before any other quantity
// could be converted to internal units.
// Therefore, we now choose fixed internal units so that these values are
// of the order of 1, but not necessarily exactly 1.

// Author: Arne Vansteenkiste.

import ()

// Primary internal units
const (
	UnitLength = 1e-9  // m
	UnitField  = 1e6   // A/m
	UnitTime   = 1e-15 // s
	UnitEnergy = 1e-18 // J
)

// Derived internal units
const ()
