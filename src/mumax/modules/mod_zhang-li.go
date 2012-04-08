//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Module implementing Slonczewski spin transfer torque.
// Authors: Graham Rowlands, Arne Vansteenkiste

import (
	. "mumax/common"
	. "mumax/engine"
	"mumax/gpu"
)

// Register this module
func init() {
	RegisterModule("zhang-li", "Zhang-Li spin transfer torque.", LoadZhangLiMADTorque)
}

func LoadZhangLiMADTorque(e *Engine) {
	e.LoadModule("llg") // needed for alpha, hfield, ...

	// ============ New Quantities =============
	e.AddNewQuant("ee", SCALAR, VALUE, Unit(""), "Degree of non-adiabadicity")
	e.AddNewQuant("pol", SCALAR, VALUE, Unit(""), "Polarization degree of the spin-current")
	LoadUserDefinedCurrentDensity(e)
	stt := e.AddNewQuant("stt", VECTOR, FIELD, Unit("/s"), "Zhang-Li Spin Transfer Torque")

	// ============ Dependencies =============
	e.Depends("stt", "ee", "pol", "j", "m")

	// ============ Updating the torque =============
	stt.SetUpdater(&ZhangLiUpdater{stt: stt})

	// Add spin-torque to LLG torque
	AddTermToQuant(e.Quant("torque"), stt)
}

type ZhangLiUpdater struct {
	stt *Quant
}

func (u *ZhangLiUpdater) Update() {
	e := GetEngine()
	
	cellSize := e.CellSize()	
	sizeMesh := e.GridSize()
	stt := u.stt
	m := e.Quant("m")
	ee := e.Quant("ee").Scalar()
	pol := e.Quant("pol").Scalar()
	curr := e.Quant("j")
	pred := pol * MuB / (E * (1 + ee * ee))
	pret := ee * pred
	
	gpu.LLZhangLi(stt.Array(), m.Array(), curr.Array(), float32(pred), float32(pret), int32(sizeMesh[X]), int32(sizeMesh[Y]), int32(sizeMesh[Z]), float32(cellSize[X]), float32(cellSize[X]), float32(cellSize[X]))
}
