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
	RegisterModule("llbr/nonlocal", "LLBr nonlocal relaxation term", LoadBaryakhtarNonLocal)
}

func LoadBaryakhtarNonLocal(e *Engine) {

	LoadHField(e)
	LoadFullMagnetization(e)
	LoadGammaLL(e)

	// ============ New Quantities =============
	e.AddNewQuant("lambda_e", VECTOR, MASK, Unit(""), "LLBr nonlocal relaxation constant")
	llbr_nonlocal := e.AddNewQuant("llbr_nonlocal", VECTOR, FIELD, Unit("/s"), "Landau-Lifshits-Baryakhtar nonlocal relaxation term")

	// ============ Dependencies =============
	e.Depends("llbr_nonlocal", "H_eff", "gamma_LL", "lambda_e", "msat0T0")

	// ============ Updating the torque =============
	upd := &BaryakhtarNonlocalUpdater{llbr_nonlocal: llbr_nonlocal}
	llbr_nonlocal.SetUpdater(upd)
}

type BaryakhtarNonlocalUpdater struct {
	llbr_nonlocal *Quant
}

func (u *BaryakhtarNonlocalUpdater) Update() {

	e := GetEngine()
	llbr_nonlocal := u.llbr_nonlocal
	gammaLL := e.Quant("gamma_LL").Scalar()
	cellSize := e.CellSize()
	heff := e.Quant("H_eff")
	pbc := e.Periodic()

	// put gamma in multiplier to avoid additional multiplications
	multiplierBT := llbr_nonlocal.Multiplier()
	for i := range multiplierBT {
		multiplierBT[i] = gammaLL
	}

	lambda_e := e.Quant("lambda_e")
	msat0T0 := e.Quant("msat0T0")

	gpu.BaryakhtarNonlocalAsync(llbr_nonlocal.Array(),
		heff.Array(),
		msat0T0.Array(),
		lambda_e.Array(),
		lambda_e.Multiplier(),
		float32(cellSize[X]),
		float32(cellSize[Y]),
		float32(cellSize[Z]),
		pbc)

	llbr_nonlocal.Array().Sync()
}
