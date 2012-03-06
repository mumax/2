//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Module implementing Slonczewski spin transfer torque.
// Authors: Graham Rowlands, Arne Vansteenkiste

import (
	. "mumax/engine"
	"mumax/gpu"
)

// Register this module
func init() {
	RegisterModule("slonczewski", "Slonczewski spin transfer torque.", LoadSlonczewskiTorque)
}

func LoadSlonczewskiTorque(e *Engine) {
	e.LoadModule("llg") // needed for alpha, hfield, ...

	// ============ New Quantities =============
	e.AddNewQuant("aj", SCALAR, VALUE, Unit(""), "In-Plane term")
	e.AddNewQuant("bj", SCALAR, VALUE, Unit(""), "Field-Like term")
	e.AddNewQuant("p", VECTOR, FIELD, Unit(""), "Polarization Vector")
	e.AddNewQuant("pol", SCALAR, VALUE, Unit(""), "Polarization Efficiency")
	LoadUserDefinedCurrentDensity(e)
	//e.AddNewQuant("curr", SCALAR, FIELD, Unit("A/m2"), "Current density")
	stt := e.AddNewQuant("stt", VECTOR, FIELD, Unit("/s"), "Slonczewski Spin Transfer Torque")

	// ============ Dependencies =============
	//e.Depends("stt", "aj", "bj", "p", "pol", "curr", "m", "gamma", "Msat")
	e.Depends("stt", "aj", "bj", "p", "pol", "j", "m", "gamma", "Msat")

	// ============ Updating the torque =============
	stt.SetUpdater(&slonczewskiUpdater{stt: stt})

	// Add spin-torque to LLG torque
	AddTermToQuant(e.Quant("torque"), stt)
}

type slonczewskiUpdater struct {
	stt *Quant
}

func (u *slonczewskiUpdater) Update() {
	e := GetEngine()
	stt := u.stt
	m := e.Quant("m")
	alpha := e.Quant("alpha")
	aj := e.Quant("aj").Scalar()
	bj := e.Quant("bj").Scalar()
	p := e.Quant("p")
	pol := e.Quant("pol").Scalar()
	curr := e.Quant("j")
	gamma := e.Quant("gamma").Scalar()
	msat := e.Quant("Msat")

	gpu.LLSlon(stt.Array(), m.Array(), p.Array(), alpha.Array(), msat.Array(),
		float32(gamma), float32(aj), float32(bj), float32(pol), curr.Array())
}
