//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// This file implements the Landau-Lifshitz torque Quantity.
// Author: Arne Vansteenkiste

import (
	. "mumax/common"
	"mumax/gpu"
)

// The reduced Landau-Lifshitz torque τ, in units gamma0*Msat:
//	d m / d t = gamma0 * Msat * τ  
// Note: the unit of gamma0 * Msat is 1/time.
// Thus:
//	τ = (m x h) - α m  x (m x h)
// with:
//	h = H / Msat
func (e *Engine) AddTorqueNode() {
	e.AddQuant("torque", VECTOR, FIELD)
	e.Depends("torque", "m", "h", "alpha", "Msat")
	t := e.Quant("torque")
	m := e.Quant("m")
	H := e.Quant("h")
	alpha := e.Quant("alpha")
	Msat := e.Quant("alpha")

	t.updater = &torqueUpdater{t, m, H, alpha, Msat}
}

type torqueUpdater struct {
	τ, m, h, α, Msat *Quant
}

func (u *torqueUpdater) Update() {
	Debug("gpu.Torque", u.τ.Array(), u.m.Array(), u.h.Array(), u.α.Array(), float32(u.α.Scalar()))
	gpu.Torque(u.τ.Array(), u.m.Array(), u.h.Array(), u.α.Array(), float32(u.α.Scalar()))
}
