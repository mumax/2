//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Module implements conservative second-order local damping
// Authors: Mykola Dvornik

import (
	. "mumax/engine"
	"mumax/gpu"
)

// Register this module
func init() {
	RegisterModule("llbar/conservative/local_02", "LLBar conservative second-order local relaxation term", LoadLLBarLocal02C)
}

func LoadLLBarLocal02C(e *Engine) {

	LoadHField(e)
	LoadFullMagnetization(e)
	LoadGammaLL(e)

	// =========== New Quantities =============

	e.AddNewQuant("μ02c", VECTOR, MASK, Unit(""), "LLBr ferromagnetic relaxation constant")
	llbr_local02c := e.AddNewQuant("llbr_local02c", VECTOR, FIELD, Unit("/s"), "Landau-Lifshits-Baryakhtar conservative second-order local relaxation term")

	// ============ Dependencies =============
	e.Depends("llbr_local02c", "mf", "H_eff", "gamma_LL", "μ02c", "msat0T0")

	// ============ Updating the torque =============
	upd := &LLBarLocal02CUpdater{llbr_local02c: llbr_local02c}
	llbr_local02c.SetUpdater(upd)
}

type LLBarLocal02CUpdater struct {
	llbr_local02c *Quant
}

func (u *LLBarLocal02CUpdater) Update() {

	e := GetEngine()

	llbr_local02c := u.llbr_local02c
	gammaLL := e.Quant("gamma_LL").Scalar()
	m := e.Quant("mf") // mf is M/Ms(T=0)
	heff := e.Quant("H_eff")

	// put gamma in multiplier to avoid additional multiplications
	multiplierBT := llbr_local02c.Multiplier()
	for i := range multiplierBT {
		multiplierBT[i] = gammaLL
	}

	mu := e.Quant("mu")
	msat0T0 := e.Quant("msat0T0")

	gpu.BaryakhtarLocal02CAsync(llbr_local02c.Array(),
		m.Array(),
		heff.Array(),
		msat0T0.Array(),
		mu.Array(),
		mu.Multiplier())

	llbr_local02c.Array().Sync()
}
