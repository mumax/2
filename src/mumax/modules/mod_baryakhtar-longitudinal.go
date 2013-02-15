//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Module implementing transverse and longitudinal Baryakhtar's' torques.
// Authors: Mykola Dvornik, Arne Vansteenkiste

import (
	. "mumax/common"
	. "mumax/engine"
	"mumax/gpu"
	//"math"
)

// Register this module

func init() {
	RegisterModule("llbr/longitudinal", "LLBr longitudinal relaxation term", LoadBaryakhtarLongitudinal)
}

func LoadBaryakhtarLongitudinal(e *Engine) {

	LoadHField(e)
	LoadFullMagnetization(e)
	LoadGammaLL(e)
	// ============ New Quantities =============

	e.AddNewQuant("lambda", SYMMTENS, MASK, Unit(""), "LLBr longitudinal relaxation constant")

	llbr_long := e.AddNewQuant("llbr_long", VECTOR, FIELD, Unit("/s"), "Landau-Lifshits-Baryakhtar longitudinal relaxation term")

	// =============== Dependencies =============
	e.Depends("llbr_long", "H_eff", "gamma_LL", "lambda", "msat0T0")

	// ============ Updating the torque =============
	upd := &BaryakhtarLongitudinalUpdater{llbr_long: llbr_long}
	llbr_long.SetUpdater(upd)
	//~ AddTermToQuant(e.Quant("llbr_RHS"), llbr_long)
}

type BaryakhtarLongitudinalUpdater struct {
	llbr_long *Quant
}

func (u *BaryakhtarLongitudinalUpdater) Update() {

	e := GetEngine()
	llbr_long := u.llbr_long
	gammaLL := e.Quant("gamma_LL").Scalar()
	heff := e.Quant("H_eff")

	// put gamma in multiplier to avoid additional multiplications
	multiplierBT := llbr_long.Multiplier()
	for i := range multiplierBT {
		multiplierBT[i] = gammaLL
	}

	lambda := e.Quant("lambda")
	msat0T0 := e.Quant("msat0T0")

	gpu.BaryakhtarLongitudinalAsync(llbr_long.Array(),
		heff.Array(),
		msat0T0.Array(),
		lambda.Array(),
		lambda.Multiplier())
		
	llbr_long.Array().Sync()
}
