//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Provides electrical current density
// Author: Arne Vansteenkiste

import (
	. "mumax/engine"
	"mumax/gpu"
)

// Register this module
func init() {
	RegisterModule("current", "Electrical currents", LoadCurrent)
}

func LoadCurrent(e *Engine) {
	if e.HasQuant("j") {
		return
	}
	LoadCoulomb(e)
	E := e.Quant("E")
	j := e.AddNewQuant("j", VECTOR, FIELD, Unit("A/m2"), "electrical current density")
	r := e.AddNewQuant("r", SCALAR, MASK, Unit("Ohm*m"), "electrical resistivity")
	e.AddNewQuant("diff_rho", SCALAR, FIELD, Unit("C/m³s"), "time derivative of electrical charge density")

	e.Depends("diff_rho", "j")
	e.Depends("j", "E", "r")
	e.AddPDE1("rho", "diff_rho")

	j.SetUpdater(&JUpdater{j: j, E: E, r: r})
}


type JUpdater struct {
	j, E, r *Quant
}


func (u *JUpdater) Update() {
	j := u.j.Array()
	gpu.CurrentDensityAsync(j, u.E.Array(), u.r.Array(), u.r.Multiplier()[0], j.Stream)
	j.Stream.Sync()
}
