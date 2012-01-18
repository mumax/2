//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Module for Faraday's law.
// Author: Arne Vansteenkiste

import (
	. "mumax/engine"
)

// Register this module
func init() {
	RegisterModule("faraday", "Faraday's law", LoadFaraday)
}

// Load Faraday's law
func LoadFaraday(e *Engine) {

	LoadEField(e)
	LoadBField(e)
	LoadDerivative(e.Quant("B"))
	LoadDerivative(e.Quant("E"))

}
//
//	// here be dragons
//	const CPUONLY = true
//	const GPU = false
//
//	// electrostatic kernel
//	kernelSize := padSize(e.GridSize(), e.Periodic())
//	elKern := NewQuant("kern_el", VECTOR, kernelSize, FIELD, Unit(""), CPUONLY, "reduced electrostatic kernel")
//	e.AddQuant(elKern)
//	elKern.SetUpdater(newElKernUpdater(elKern))
//
//	e.Depends("E", "rho", "kern_el")
//}
//
//// Updates electrostatic kernel (cpu)
//type elKernUpdater struct {
//	kern *Quant // that's me!
//}
//
//func newElKernUpdater(elKern *Quant) Updater {
//	u := new(elKernUpdater)
//	u.kern = elKern
//	return u
//}
//
//// Update electrostatic kernel (cpu)
//func (u *elKernUpdater) Update() {
//	e := GetEngine()
//
//	// first update the kernel
//	kernsize := padSize(e.GridSize(), e.Periodic())
//	Log("Calculating electrosatic kernel, may take a moment...")
//	PointKernel(kernsize, e.CellSize(), e.Periodic(), u.kern.Buffer())
//
//	// then also load it into the E field convolution
//	EUpdater := e.Quant("E").GetUpdater().(*EfieldUpdater)
//	kernEl := GetEngine().Quant("kern_el").Buffer()
//	EUpdater.conv.LoadKernel(kernEl, 0, gpu.DIAGONAL, gpu.PUREIMAG)
//	EUpdater.convInput[0] = e.Quant("rho").Array()
//}
