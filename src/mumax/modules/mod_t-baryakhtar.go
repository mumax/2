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
	RegisterModule("baryakhtar", "Baryakhtar's relaxation term", LoadBaryakhtarTorques)
}

func LoadBaryakhtarTorques(e *Engine) {
	e.LoadModule("llg") // needed for alpha, hfield, ...
    e.LoadModule("longfield") // needed for initial distribution of satruration magnetization
    LoadHField(e)
    LoadMagnetization(e)
	// ============ New Quantities =============

	e.AddNewQuant("lambda", SCALAR, VALUE, Unit("A/m"), "Landau-Lifshits relaxation constant")
	e.AddNewQuant("lambda_e", SCALAR, VALUE, Unit("A/m"), "Baryakhtar's exchange relaxation constant")
	e.AddNewQuant("gamma_LL", SCALAR, VALUE, Unit("m/As"), "Landau-Lifshits gyromagetic ratio")
	//e.AddNewQuant("debug_h", VECTOR, FIELD, Unit("A/m"), "Debug effective field to check laplacian implementation")
	bdt := e.AddNewQuant("bdt", VECTOR, FIELD, Unit("/s"), "Baryakhtar's perpendicular relaxation term")
    bdl := e.AddNewQuant("bdl", SCALAR, FIELD, Unit("/s"), "Baryakhtar's longitudinal relaxation term")
	// ============ Dependencies =============
	e.Depends("bdt", "beta", "m", "msat0", "Aex", "H_eff", "gamma", "alpha", "blambda")
    e.Depends("bdl", "beta", "m", "msat0", "Aex", "H_eff", "gamma", "alpha", "blambda")
    
	// ============ Updating the torque =============
	upd := &BaryakhtarUpdater{bdt: bdt, bdl: bdl}
	bdt.SetUpdater(upd)
    bdl.SetUpdater(upd)
}

type BaryakhtarUpdater struct {
	bdt, bdl *Quant
}

func (u *BaryakhtarUpdater) Update() {

	e := GetEngine()	
	cellSize := e.CellSize()	
	bdt := u.bdt
	bdl := u.bdt
	m := e.Quant("m")
	beta := e.Quant("beta").Scalar()
	msat := e.Quant("msat0") // it is pointwise
	heff := e.Quant("H_eff")
	aex := e.Quant("Aex")
	gamma := e.Quant("gamma").Scalar()
	alpha := e.Quant("alpha")
	pbc := e.Periodic()
	//debug_h := e.Quant("debug_h")
	
	pre := beta * naexn / (Mu0 * nmsatn * nmsatn) 
	pred := gamma * pre 
	
	gpu.LLGBtAsync(bdt.Array(), 
                    bdl.Array(),
                    m.Array(), 
                    heff.Array(), 
                    msat.Array(), 
                    aex.Array(), 
                    alpha.Array(),
                    float32(alpha.Multiplier()[0]),
                    float32(pred), 
                    float32(pre),
                    float32(blambda),
                    float32(cellSize[X]), 
                    float32(cellSize[Y]), 
                    float32(cellSize[Z]), 
                    pbc)
                    
    bdt.Array().Sync()
    
    bdt.SetUpToDate(true)
    bdl.SetUpToDate(true)
}
