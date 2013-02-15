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
	RegisterModule("llbr/torque", "LLBr torque term", LoadBaryakhtarTorque)
}

func LoadBaryakhtarTorque(e *Engine) {
	
	LoadHField(e)
	LoadFullMagnetization(e)
	LoadGammaLL(e)
	
	// ============ New Quantities =============
	
	llbr_torque := e.AddNewQuant("llbr_torque", VECTOR, FIELD, Unit("/s"), "Landau-Lifshits-Baryakhtar torque")

	// ============ Dependencies =============
	e.Depends("llbr_torque", "mf", "H_eff", "gamma_LL")

	// ============ Updating the torque =============
	upd := &BaryakhtarTorqueAsyncUpdater{llbr_torque: llbr_torque}
	llbr_torque.SetUpdater(upd)
	
	//~ AddTermToQuant(e.Quant("llbr_RHS"), llbr_torque)
}

type BaryakhtarTorqueAsyncUpdater struct {
	llbr_torque *Quant
}

func (u *BaryakhtarTorqueAsyncUpdater) Update() {

	e := GetEngine()
	llbr_torque := u.llbr_torque
	gammaLL := e.Quant("gamma_LL").Scalar()
	m := e.Quant("mf")
	heff := e.Quant("H_eff")
	
	// put gamma in multiplier to avoid additional multiplications
	multiplierBT := llbr_torque.Multiplier()
	
	for i := range multiplierBT {
		multiplierBT[i] = gammaLL
	}

	msat0T0 := e.Quant("msat0T0")

	gpu.BaryakhtarTorqueAsync(llbr_torque.Array(),
		m.Array(),
		heff.Array(),
		msat0T0.Array())
	
	llbr_torque.Array().Sync()
}
