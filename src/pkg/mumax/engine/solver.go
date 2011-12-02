//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

import (
//. "mumax/common"
)

// A Solver is a method to advance a quantity in time.
// Multiple solvers may be active at the same time.
// Therefore all solvers are first asked to AdvanceBuffer(),
// which time-steps but hides the result in a buffer. The result
// has to be hidden because other solvers may take the output quantity
// as input and they should not receive its value "from the future".
// When all solvers have buffered their output, they are asked to 
// CopyBuffer(), which copies their output buffer to the output quantity.
// TODO: the engine could check the need for CopyBuffer()
type Solver interface {
	AdvanceBuffer()           // Takes one time step but hides the result in a buffer
	CopyBuffer()              // Overwrite the output quantity with the previously calculated buffer
	ProposeDt() float64       // Propose a new time step based on error estimate. May be 0 (ignored)
	Deps() (in, out []*Quant) // Input/Output quantities (not including time step dt)
}
