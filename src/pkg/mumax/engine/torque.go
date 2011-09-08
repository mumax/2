//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// This file implements the Landau-Lifshitz torque Quantity
// Author: Arne Vansteenkiste

import (
	"mumax/gpu"
)

func (e *Engine) AddTorqueNode() {
	e.AddQuant("torque", VECTOR, FIELD)
	e.Depends("torque", "m", "H", "alpha")
	t := e.Quant("torque")
	m := e.Quant("m")
	H := e.Quant("H")
	alpha := e.Quant("alpha")

	t.updateSelf = &torqueUpdater{t, m, H, alpha}
}

type torqueUpdater struct {
	τ, m, h, α *Quant
}

func (u *torqueUpdater) Update() {
	gpu.Torque(u.τ.Array(), u.m.Array(), u.h.Array(), u.α.Array(), u.α.multiplier[0])
}
