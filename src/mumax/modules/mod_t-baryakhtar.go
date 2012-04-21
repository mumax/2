//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Module implementing Slonczewski spin transfer torque.
// Authors: Mykola Dvornik, Arne Vansteenkiste

import (
	. "mumax/common"
	. "mumax/engine"
	"mumax/gpu"
	//"math"
)

// Register this module
func init() {
	RegisterModule("t-baryakhtar", "Baryakhtar's perpendicular relaxation term", LoadTBaryakhtarDamp)
}

func LoadTBaryakhtarDamp(e *Engine) {
	e.LoadModule("llg") // needed for alpha, hfield, ...

	// ============ New Quantities =============
	e.AddNewQuant("beta", SCALAR, VALUE, Unit(""), "Perpendicular Relaxation Rate")
	bdt := e.AddNewQuant("bdt", VECTOR, FIELD, Unit("/s"), "Baryakhtar's perpendicular relaxation term")

	// ============ Dependencies =============
	e.Depends("bdt", "beta", "m", "msat", "Aex", "H_eff")

	// ============ Updating the torque =============
	bdt.SetUpdater(&TBaryakhtarUpdater{bdt: bdt})

	// Add spin-torque to LLG torque
	AddTermToQuant(e.Quant("torque"), bdt)
}

type TBaryakhtarUpdater struct {
	bdt *Quant
}

func (u *TBaryakhtarUpdater) Update() {
	e := GetEngine()
	
	cellSize := e.CellSize()	
	bdt := u.bdt
	m := e.Quant("m")
	beta := e.Quant("beta").Scalar()
	msat := e.Quant("msat") // it is pointwise
	heff := e.Quant("H_eff")
	aex := e.Quant("Aex")
	pbc := e.Periodic()
	
	nmsatn := msat.Multiplier()[0]
	
	pred := beta * aex.Multiplier()[0] / (Mu0 * nmsatn * nmsatn) 
	
	gpu.LLGBt(bdt.Array(), m.Array(), heff.Array(), msat.Array(), aex.Array(), float32(pred), float32(cellSize[X]), float32(cellSize[Y]), float32(cellSize[Z]), pbc)
}
