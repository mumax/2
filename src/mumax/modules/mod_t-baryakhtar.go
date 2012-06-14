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
    LoadMagnetization(e)
    LoadFullMagnetization(e)
	// ============ New Quantities =============

	e.AddNewQuant("lambda", SCALAR, VALUE, Unit("A/m"), "Landau-Lifshits relaxation constant")
	e.AddNewQuant("lambda_e", SCALAR, VALUE, Unit("A/m"), "Baryakhtar's exchange relaxation constant")
	e.AddNewQuant("gamma_LL", SCALAR, VALUE, Unit("m/As"), "Landau-Lifshits gyromagetic ratio")
	//e.AddNewQuant("debug_h", VECTOR, FIELD, Unit("A/m"), "Debug effective field to check laplacian implementation")
	btorque := e.AddNewQuant("btorque", VECTOR, FIELD, Unit("/s"), "Landau-Lifshits torque plus Baryakhtar relaxation")
	
	// ============ Dependencies =============
	e.Depends("btorque", "mf", "H_eff", "gamma_LL", "lambda", "lambda_e","msat0");//,"debug_h")
    
	// ============ Updating the torque =============
	upd := &BaryakhtarUpdater{btorque: btorque}
	btorque.SetUpdater(upd)
}

type BaryakhtarUpdater struct {
	btorque *Quant
}

func (u *BaryakhtarUpdater) Update() {

	e := GetEngine()	
	cellSize := e.CellSize()	
	btorque := u.btorque
	m := e.Quant("mf")
	lambda := e.Quant("lambda").Scalar()
    lambda_e := e.Quant("lambda_e").Scalar()
	heff := e.Quant("H_eff")
	gammaLL := e.Quant("gamma_LL").Scalar()	
	pbc := e.Periodic()
	msat0 := e.Quant("msat0")
	//debug_h := e.Quant("debug_h")
	
	// put lambda in multiplier to avoid additional multiplications
	multiplierBT := btorque.Multiplier()
	for i := range multiplierBT {
		multiplierBT[i] = gammaLL
	}
	
	gpu.LLGBtAsync(btorque.Array(), 
                    m.Array(), 
                    heff.Array(),
                    msat0.Array(),
                    float32(msat0.Multiplier()[0]),
                    float32(lambda), 
                    float32(lambda_e),
                    float32(cellSize[X]), 
                    float32(cellSize[Y]), 
                    float32(cellSize[Z]), 
                    pbc)
                    
    btorque.Array().Sync()
}
