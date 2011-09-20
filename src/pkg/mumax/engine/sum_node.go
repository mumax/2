//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// This file implements the sum of Quantities.
// Author: Arne Vansteenkiste

import (
	. "mumax/common"
	"mumax/gpu"
)

func (e *Engine) AddSumNode(name string, args ...string) {
	parent0 := e.Quant(args[0])
	nComp := parent0.NComp()
	unit := parent0.Unit()
	e.AddQuant(name, nComp, FIELD, unit)

	sum := e.Quant(name)
	parents := make([]*Quant, len(args))
	for i := range parents {
		parents[i] = e.Quant(args[i])
		if parents[i].Unit() != sum.Unit() {
			panic(InputErr("sum: mismatched units: " + sum.FullName() + " <-> " + parents[i].FullName()))
		}
	}
	e.Depends(name, args...)
	sum.updater = &sumUpdater{sum, parents}
}

type sumUpdater struct {
	sum     *Quant
	parents []*Quant
}

func (u *sumUpdater) Update() {
	// TODO: optimize for 0,1,2 or more parents
	sum := u.sum
	sum.array.Zero()
	parents := u.parents
	for i := range parents {
		parent := parents[i]
		for c := 0; c < sum.NComp(); c++ {
			parComp := parent.array.Component(c)
			parMul := parent.multiplier[c]
			sumMul := sum.multiplier[c]
			sumComp := sum.array.Component(c)
			//Debug("gpu.Madd", sumComp, sumComp, parComp, float32(parMul))
			gpu.Madd(sumComp, sumComp, parComp, float32(parMul/sumMul)) // divide by sum's multiplier!
		}
	}
}
