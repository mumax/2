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

//func  AddSumNode(name string, args ...string) {
//		e := GetEngine()
//	parent0 := e.Quant(args[0])
//	nComp := parent0.NComp()
//	unit := parent0.Unit()
//	e.AddQuant(name, nComp, FIELD, unit)
//
//	sum := e.Quant(name)
//	parents := make([]*Quant, len(args))
//	for i := range parents {
//		parents[i] = e.Quant(args[i])
//		if parents[i].Unit() != sum.Unit() {
//			panic(InputErr("sum: mismatched units: " + sum.FullName() + " <-> " + parents[i].FullName()))
//		}
//	}
//	e.Depends(name, args...)
//	sum.updater = &SumUpdater{sum, parents}
//}

type SumUpdater struct {
	sum     *Quant
	//parents []*Quant
}

func (u *SumUpdater) Update() {
	// TODO: optimize for 0,1,2 or more parents
	sum := u.sum
	sum.array.Zero()
	parents := u.sum.parents
	for i := range parents {
		parent := parents[i]
		for c := 0; c < sum.NComp(); c++ {
			parComp := parent.array.Component(c)
			parMul := parent.multiplier[c]
			sumMul := sum.multiplier[c]
			sumComp := sum.array.Component(c)
			Debug("gpu.Madd",sumComp, sumComp, parComp, float32(parMul/sumMul))
			gpu.Madd(sumComp, sumComp, parComp, float32(parMul/sumMul)) // divide by sum's multiplier!
		}
	}
}

// Adds a parent to the sum, i.e., its value will be added to the sum
func (u *SumUpdater) AddParent(name string) {
	e := GetEngine()
	parent := e.Quant(name)
	sum := u.sum
	if parent.unit != sum.unit {
		panic(InputErr("sum: mismatched units: " + sum.FullName() + " <-> " + parent.FullName()))
	}
	//u.parents = append(u.parents, parent) done by engine
	e.Depends(sum.Name(), name)
}
