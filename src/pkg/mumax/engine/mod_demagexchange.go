//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// Combined demag+exchange module
// Author: Arne Vansteenkiste

import (
	"mumax/gpu"
)

// Register this module
func init() {
	RegisterModule(&ModDemagExch{})
}

// Module for combined calculation of demag + exchange field
// in one single convolution.
type ModDemagExch struct{}

func (x ModDemagExch) Description() string {
	return "combined magnetostatic + exchange field"
}

func (x ModDemagExch) Name() string {
	return "demagexch"
}

func (x ModDemagExch) Load(e *Engine) {
	// dependencies
	e.LoadModule("hfield")
	e.LoadModule("magnetization")
	e.LoadModule("aexchange")

	// demag+exchange field quant
	e.AddQuant("H_dex", VECTOR, FIELD, Unit("A/m"), "demag+exchange field")
	e.Depends("H_dex", "Aex", "m", "Msat")
	Hdex := e.Quant("H_dex")
	Hdex.updater = newDemagExchUpdater(Hdex, e.Quant("m"), e.Quant("Msat"), e.Quant("Aex"))

	// add H_dex to total H
	hfield := e.Quant("H")
	sum := hfield.updater.(*SumUpdater)
	sum.AddParent("H_dex")
}

// Updates the demag+exchange field in one single convolution
type demagExchUpdater struct {
	Hdex, m, Msat, Aex *Quant
	conv               gpu.ConvPlan // TODO: move gpu.ConvPlan into engine?
}

func (u *demagExchUpdater) Update() {

}

func newDemagExchUpdater(Hdex, m, Msat, Aex *Quant) Updater {
	u := new(demagExchUpdater)
	u.Hdex = Hdex
	u.m = m
	u.Msat = Msat
	u.Aex = Aex

	e := GetEngine()
	kernsize := padSize(e.GridSize(), e.Periodic())
	accuracy := 8
	kernel := FaceKernel6(kernsize, e.CellSize(), accuracy, e.Periodic())

	u.conv.Init(e.GridSize(), kernel)
	return u
}

// Zero-pads gridsize if needed
// (not periodic, not size 1)
func zeropad(gridsize, periodic []int) (padded []int) {
	padded = make([]int, 3)
	for i := range gridsize {
		if gridsize[i] > 1 && periodic[i] == 0 {
			padded[i] = 2 * gridsize[i]
		} else {
			padded[i] = gridsize[i]
		}
	}
	return padded
}
