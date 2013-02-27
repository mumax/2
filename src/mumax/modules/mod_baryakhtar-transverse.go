//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Module implementing transverse and longitudinal Baryakhtar's' torques.
// Authors: Mykola Dvornik

import (
	//~ . "mumax/common"
	. "mumax/engine"
	"mumax/gpu"
	//"math"
)

// Register this module
func init() {
	RegisterModule("llbr/transverse", "LLBr transverse relaxation term", LoadBaryakhtarTransverse)
}

func LoadBaryakhtarTransverse(e *Engine) {

	e.LoadModule("longfield")
	LoadHField(e)
	LoadFullMagnetization(e)
	LoadGammaLL(e)
	
	// =========== New Quantities =============

	e.AddNewQuant("mu", SYMMTENS, MASK, Unit(""), "LLBr transverse relaxation constant")
	llbr_transverse := e.AddNewQuant("llbr_transverse", VECTOR, FIELD, Unit("/s"), "Landau-Lifshits-Baryakhtar transverse relaxation term")
	
	// ============ Dependencies =============
	e.Depends("llbr_transverse", "mf", "H_eff", "gamma_LL", "mu", "msat0T0")

	// ============ Updating the torque =============
	upd := &BaryakhtarTransverseUpdater{llbr_transverse: llbr_transverse}
	llbr_transverse.SetUpdater(upd)
	//~ AddTermToQuant(e.Quant("llbr_RHS"), llbr_transverse)
	
}

type BaryakhtarTransverseUpdater struct {
	llbr_transverse *Quant
}

func (u *BaryakhtarTransverseUpdater) Update() {

	e := GetEngine()
	
	llbr_transverse := u.llbr_transverse
	gammaLL := e.Quant("gamma_LL").Scalar()
	m := e.Quant("mf")
	heff := e.Quant("H_eff")

	// put gamma in multiplier to avoid additional multiplications
	multiplierBT := llbr_transverse.Multiplier()
	for i := range multiplierBT {
		multiplierBT[i] = gammaLL
	}

	mu := e.Quant("mu")
	msat0T0 := e.Quant("msat0T0")

	gpu.BaryakhtarTransverseAsync(llbr_transverse.Array(),
		m.Array(),
		heff.Array(),
		msat0T0.Array(),
		mu.Array(),
		mu.Multiplier())
		
	llbr_transverse.Array().Sync()
}
