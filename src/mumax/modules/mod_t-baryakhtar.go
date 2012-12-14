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
	RegisterModule("baryakhtar", "Baryakhtar-Ivanov relaxation term", LoadBaryakhtarTorques)
}

func LoadBaryakhtarTorques(e *Engine) {

	e.LoadModule("longfield") // needed for initial distribution of satruration magnetization
	LoadHField(e)
	LoadFullMagnetization(e)
	LoadGammaLL(e)
	// ============ New Quantities =============

	e.AddNewQuant("lambda", SYMMTENS, MASK, Unit(""), "Landau-Lifshits relaxation constant")
	e.AddNewQuant("lambda_e", SYMMTENS, MASK, Unit(""), "Baryakhtar's exchange relaxation constant")
	//e.AddNewQuant("debug_h", VECTOR, FIELD, Unit("A/m"), "Debug effective field to check laplacian implementation")
	btorque := e.AddNewQuant("torque", VECTOR, FIELD, Unit("/s"), "Landau-Lifshits torque plus Baryakhtar relaxation")

	// ============ Dependencies =============
	e.Depends("torque", "mf", "H_eff", "gamma_LL", "lambda", "lambda_e", "msat0T0") //, "debug_h")

	// ============ Updating the torque =============
	upd := &BaryakhtarUpdater{btorque: btorque}
	btorque.SetUpdater(upd)
}

type BaryakhtarUpdater struct {
	btorque *Quant
}

func (u *BaryakhtarUpdater) Update() {

	e := GetEngine()
	btorque := u.btorque
	gammaLL := e.Quant("gamma_LL").Scalar()
	cellSize := e.CellSize()
	m := e.Quant("mf")
	heff := e.Quant("H_eff")
	pbc := e.Periodic()
	//heff := e.Quant("debug_h")

	// put gamma in multiplier to avoid additional multiplications
	multiplierBT := btorque.Multiplier()
	for i := range multiplierBT {
		multiplierBT[i] = gammaLL
	}

	lambda := e.Quant("lambda")
	lambda_e := e.Quant("lambda_e")
	msat0T0 := e.Quant("msat0T0")

	//Debug(lambda.Multiplier()[XX], lambda.Multiplier()[YY], lambda.Multiplier()[ZZ], lambda.Multiplier()[XY], lambda.Multiplier()[XZ], lambda.Multiplier()[YZ])
	//Debug(lambda_e.Multiplier()[XX], lambda_e.Multiplier()[YY], lambda_e.Multiplier()[ZZ], lambda_e.Multiplier()[XY], lambda_e.Multiplier()[XZ], lambda_e.Multiplier()[YZ])

	gpu.LLGBtAsync(btorque.Array(),
		m.Array(),
		heff.Array(),
		msat0T0.Array(),
		lambda.Array(),
		lambda_e.Array(),
		lambda.Multiplier(),
		lambda_e.Multiplier(),
		float32(cellSize[X]),
		float32(cellSize[Y]),
		float32(cellSize[Z]),
		pbc)

	btorque.Array().Sync()
}
