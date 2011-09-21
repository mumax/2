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

func (e *Engine) Step() {
	//Debug("Engine.Step")

	// update input for ODE solver recursively
	for _, ode := range e.ode {
		ode[RHS].Update()
	}

	// step
	for _, solver := range e.solver {
		solver.Step()
	}
	e.time.SetScalar(e.time.Scalar() + e.dt.Scalar())

	// set new t, dt, m

	// invalidate everything that depends on solver
	e.dt.Invalidate()
	e.time.Invalidate()
	for _, ode := range e.ode {
		ode[LHS].Invalidate()
	}

}
