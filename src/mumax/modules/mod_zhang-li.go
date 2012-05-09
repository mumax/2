//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Module implementing Slonczewski spin transfer torque.
// Authors: Mykola Dvornik, Arne Vansteenkiste

import (
	. "mumax/common"
	. "mumax/engine"
	"mumax/gpu"
	//"math"
)

// Register this module
func init() {
	RegisterModule("zhang-li", "Zhang-Li spin transfer torque.", LoadZhangLiMADTorque)
}

func LoadZhangLiMADTorque(e *Engine) {
	e.LoadModule("llg") // needed for alpha, hfield, ...

	// ============ New Quantities =============
	xi := e.AddNewQuant("xi", SCALAR, VALUE, Unit(""), "Degree of non-adiabadicity")
	xi.SetScalar(0.05)
	pol := e.AddNewQuant("polarisation", SCALAR, VALUE, Unit(""), "Polarization degree of the spin-current")
	pol.SetScalar(1.0)
	LoadUserDefinedCurrentDensity(e)
	zzt := e.AddNewQuant("zzt", VECTOR, FIELD, Unit("/s"), "Zhang-Li Spin Transfer Torque")

	// ============ Dependencies =============
	e.Depends("zzt", "xi", "polarisation", "j", "m", "msat")

	// ============ Updating the torque =============
	zzt.SetUpdater(&ZhangLiUpdater{zzt: zzt})

	// Add spin-torque to LLG torque
	AddTermToQuant(e.Quant("torque"), zzt)
}

type ZhangLiUpdater struct {
	zzt *Quant
}

func (u *ZhangLiUpdater) Update() {
	e := GetEngine()
	
	cellSize := e.CellSize()	
	zzt := u.zzt
	m := e.Quant("m")
	ee := e.Quant("xi").Scalar()
	msat := e.Quant("msat") // it is pointwise
	pol := e.Quant("polarisation").Scalar()
	curr := e.Quant("j") // could be pointwise
	pbc := e.Periodic()
	
	//njn := math.Sqrt(float64(curr.Multiplier()[0] * curr.Multiplier()[0]) + float64(curr.Multiplier()[1] * curr.Multiplier()[1]) + float64(curr.Multiplier()[2] * curr.Multiplier()[2]))
	nmsatn := msat.Multiplier()[0]
	
	pred := pol * MuB / (E * nmsatn * (1 + ee * ee)) 
	pret := ee * pred
	
	gpu.LLZhangLi(zzt.Array(), m.Array(), curr.Array(), msat.Array(), curr.Multiplier(), float32(pred), float32(pret), float32(cellSize[X]), float32(cellSize[Y]), float32(cellSize[Z]), pbc)
}
