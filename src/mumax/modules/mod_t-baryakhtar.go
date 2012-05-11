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
	// ============ New Quantities =============

	e.AddNewQuant("beta", SCALAR, VALUE, Unit(""), "Baryakhtar's exchange relaxation constant")
	e.AddNewQuant("blambda", SCALAR, VALUE, Unit(""), "Baryakhtar's relativistic relaxation constant")
	
	bdt := e.AddNewQuant("bdt", VECTOR, FIELD, Unit("/s"), "Baryakhtar's perpendicular relaxation term")
    bdl := e.AddNewQuant("bdl", SCALAR, FIELD, Unit("m/As"), "Baryakhtar's longitudinal relaxation term")
    bdt.Multiplier()[0] = 1.0
    bdt.Multiplier()[1] = 1.0
    bdt.Multiplier()[2] = 1.0
    
    bdl.Multiplier()[0] = 1.0
	// ============ Dependencies =============
	e.Depends("bdt", "beta", "m", "msat0", "Aex", "H_eff", "gamma", "alpha", "blambda")
    e.Depends("bdl", "beta", "m", "msat0", "Aex", "H_eff", "gamma", "alpha", "blambda")
    
	// ============ Updating the torque =============
	upd := &BaryakhtarUpdater{bdt: bdt, bdl: bdl}
	bdt.SetUpdater(upd)
    bdl.SetUpdater(upd) 
    AddTermToQuant(e.Quant("torque"), bdt)
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
	beta := e.Quant("beta").Scalar()
	msat := e.Quant("msat0") // it is pointwise
	heff := e.Quant("H_eff")
	aex := e.Quant("Aex")
	gamma := e.Quant("gamma").Scalar()
	msat0 := e.Quant("msat0")
	alpha := e.Quant("alpha")
	pbc := e.Periodic()
	blambda := e.Quant("blambda").Scalar()
	
	nmsatn := msat.Multiplier()[0]
	naexn := aex.Multiplier()[0]
	
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
                    float32(msat0.Multiplier()[0]),
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
