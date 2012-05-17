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
	RegisterModule("baryakhtar", "Baryakhtar's-Ivanov relaxation term", LoadBaryakhtarTorques)
}

func LoadBaryakhtarTorques(e *Engine) {

    e.LoadModule("longfield") // needed for initial distribution of satruration magnetization
    LoadHField(e)
	// ============ New Quantities =============

	e.AddNewQuant("lambda", SCALAR, VALUE, Unit("A/m"), "Landau-Lifshits relaxation constant")
	e.AddNewQuant("lambda_e", SCALAR, VALUE, Unit("A/m"), "Baryakhtar's exchange relaxation constant")
	e.AddNewQuant("gamma_LL", SCALAR, VALUE, Unit("m/As"), "Landau-Lifshits gyromagetic ratio")
	bdt := e.AddNewQuant("bdt", VECTOR, FIELD, Unit("/s"), "Baryakhtar's perpendicular relaxation term")
    bdl := e.AddNewQuant("bdl", SCALAR, FIELD, Unit("/s"), "Baryakhtar's longitudinal relaxation term")
    //bdt.Multiplier()[0] = 1.0
    //bdt.Multiplier()[1] = 1.0
    //bdt.Multiplier()[2] = 1.0
    
    //bdl.Multiplier()[0] = 1.0
	// ============ Dependencies =============
	e.Depends("bdt", "m", "msat", "H_eff", "gamma_LL", "lambda", "lambda_e")
    e.Depends("bdl", "m", "msat", "H_eff", "gamma_LL", "lambda", "lambda_e")
    
	// ============ Updating the torque =============
	upd := &BaryakhtarUpdater{bdt: bdt, bdl: bdl}
	bdt.SetUpdater(upd)
    bdl.SetUpdater(upd) 
    //AddTermToQuant(e.Quant("torque"), bdt)
}

type BaryakhtarUpdater struct {
	bdt, bdl *Quant
}

func (u *BaryakhtarUpdater) Update() {

	e := GetEngine()	
	cellSize := e.CellSize()	
	bdt := u.bdt
	bdl := u.bdl
	m := e.Quant("m")
	lambda := e.Quant("lambda").Scalar()
    lambda_e := e.Quant("lambda_e").Scalar()
	msat := e.Quant("msat") // it is pointwise
	heff := e.Quant("H_eff")
	gammaLL := e.Quant("gamma_LL").Scalar()	
	pbc := e.Periodic()
	
	
	// put lambda in multiplier to avoid additional multiplications
	multiplierBDT := bdt.Multiplier()
	for i := range multiplierBDT {
		multiplierBDT[i] = gammaLL
	}
	
	multiplierBDL := bdl.Multiplier()
	for i := range multiplierBDL {
		multiplierBDL[i] = gammaLL
	}
	
	gpu.LLGBtAsync(bdt.Array(), 
                    bdl.Array(),
                    m.Array(), 
                    heff.Array(), 
                    msat.Array(),
                    float32(msat.Multiplier()[0]),
                    float32(lambda), 
                    float32(lambda_e),
                    float32(cellSize[X]), 
                    float32(cellSize[Y]), 
                    float32(cellSize[Z]), 
                    pbc)
                    
    bdt.Array().Sync()
    bdt.SetUpToDate(true)
    bdl.SetUpToDate(true)
}
