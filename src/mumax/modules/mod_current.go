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
	RegisterModule("current", "Electrical currents", LoadCalculatedCurrentDensity)
}

// loads the current density
func LoadUserDefinedCurrentDensity(e *Engine) {
	if e.HasQuant("j") {
		return
	}
	e.AddNewQuant("j", VECTOR, MASK, Unit("A/m2"), "electrical current density")
}

// calculate current density
func LoadCalculatedCurrentDensity(e *Engine) {
	if e.HasQuant("diff_rho") {
		return
	}
	LoadCoulomb(e)
	LoadUserDefinedCurrentDensity(e)
	j := e.Quant("j")
	E := e.Quant("E")
	r := e.AddNewQuant("r", SCALAR, MASK, Unit("Ohm*m"), "electrical resistivity")
	drho := e.AddNewQuant("diff_rho", SCALAR, FIELD, Unit("C/m³s"), "time derivative of electrical charge density")

	e.Depends("diff_rho", "j")
	e.Depends("j", "E", "r")
	e.AddPDE1("rho", "diff_rho")

	j.SetUpdater(&JUpdater{j: j, E: E, r: r})
	drho.SetUpdater(&DRhoUpdater{drho: drho, j: j})
}

// Updates time derivative of charge density
type DRhoUpdater struct {
	drho, j *Quant
}

func (u *DRhoUpdater) Update() {
	e := GetEngine()
	drho := u.drho.Array()
	gpu.DiffRhoAsync(drho, u.j.Array(), e.CellSize(), e.Periodic(), drho.Stream)
	drho.Stream.Sync()
}

// Updates current density
type JUpdater struct {
	j, E, r *Quant
}

func (u *JUpdater) Update() {
	e := GetEngine()
	j := u.j.Array()
	gpu.CurrentDensityAsync(j, u.E.Array(), u.r.Array(), u.r.Multiplier()[0], e.Periodic(), j.Stream)
	j.Stream.Sync()
}
