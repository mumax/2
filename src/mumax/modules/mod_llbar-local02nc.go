//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Module implements nonconservative second-order local damping
// Authors: Mykola Dvornik

import (
	. "mumax/engine"
	"mumax/gpu"
)

// Register this module
func init() {
	RegisterModule("llbar/damping/nonconservative/02/local", "LLBar nonconservative second-order local relaxation term", LoadLLBarLocal02NC)
}

func LoadLLBarLocal02NC(e *Engine) {

	LoadHField(e)
	LoadFullMagnetization(e)
	LoadGammaLL(e)

	// =========== New Quantities =============

	e.AddNewQuant("μ∥", VECTOR, MASK, Unit(""), "LLBar second-order local nonconservative relaxation diagonal tensor")
	llbar_local02nc := e.AddNewQuant("llbar_local02nc", VECTOR, FIELD, Unit("/s"), "Landau-Lifshitz-Baryakhtar nonconservative second-order local relaxation term")

	// ============ Dependencies =============
	e.Depends("llbar_local02nc", "mf", "H_eff", "γ_LL", "μ∥", "msat0T0")

	// ============ Updating the torque =============
	upd := &LLBarLocal02NCUpdater{llbar_local02nc: llbar_local02nc}
	llbar_local02nc.SetUpdater(upd)
}

type LLBarLocal02NCUpdater struct {
	llbar_local02nc *Quant
}

func (u *LLBarLocal02NCUpdater) Update() {

	e := GetEngine()

	llbar_local02nc := u.llbar_local02nc
	gammaLL := e.Quant("γ_LL").Scalar()
	m := e.Quant("mf") // mf is M/Ms(T=0)
	heff := e.Quant("H_eff")

	// put gamma in multiplier to avoid additional multiplications
	multiplierBT := llbar_local02nc.Multiplier()
	for i := range multiplierBT {
		multiplierBT[i] = gammaLL
	}

	mu := e.Quant("μ∥")
	msat0T0 := e.Quant("msat0T0")

	gpu.LLBarLocal02NC(llbar_local02nc.Array(),
		m.Array(),
		heff.Array(),
		msat0T0.Array(),
		mu.Array(),
		mu.Multiplier())

	llbar_local02nc.Array().Sync()
}
