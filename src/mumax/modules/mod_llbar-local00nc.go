//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Module implements nonconservative zero-order local damping
// Authors: Mykola Dvornik, Arne Vansteenkiste

import (
	. "mumax/engine"
	"mumax/gpu"
)

// Register this module

func init() {
	RegisterModule("llbar/damping/nonconservative/local_00", "LLBar nonconservative zero-order local relaxation term", LoadLLBarLocal00NC)
}

func LoadLLBarLocal00NC(e *Engine) {

	LoadHField(e)
	LoadFullMagnetization(e)
	LoadGammaLL(e)
	e.LoadModule("longfield")

	// ============ New Quantities =============

	e.AddNewQuant("λ∥", VECTOR, MASK, Unit(""), "LLBar zero-order local relaxation diagonal tensor")

	llbar_local00nc := e.AddNewQuant("llbar_local00nc", VECTOR, FIELD, Unit("/s"), "Landau-Lifshitz-Baryakhtar nonconservative zero-order local relaxation term")

	// =============== Dependencies =============
	e.Depends("llbar_local00nc", "H_eff", "gamma_LL", "λ∥", "msat0T0")

	// ============ Updating the torque =============
	upd := &LLBarLocal00NCUpdater{llbar_local00nc: llbar_local00nc}
	llbar_local00nc.SetUpdater(upd)
}

type LLBarLocal00NCUpdater struct {
	llbar_local00nc *Quant
}

func (u *LLBarLocal00NCUpdater) Update() {

	e := GetEngine()
	llbar_local00nc := u.llbar_local00nc
	gammaLL := e.Quant("gamma_LL").Scalar()
	heff := e.Quant("H_eff")

	// put gamma in multiplier to avoid additional multiplications
	multiplierBT := llbar_local00nc.Multiplier()
	for i := range multiplierBT {
		multiplierBT[i] = gammaLL
	}

	lambda := e.Quant("λ∥")
	msat0T0 := e.Quant("msat0T0")

	gpu.LLBarLocal00NC(llbar_local00nc.Array(),
		heff.Array(),
		msat0T0.Array(),
		lambda.Array(),
		lambda.Multiplier())

	llbar_local00nc.Array().Sync()
}
