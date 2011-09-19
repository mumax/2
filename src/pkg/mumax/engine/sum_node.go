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
	"mumax/gpu"
)

func (e *Engine) AddSumNode(name string, args ...string) {
	parent0 := e.Quant(args[0])
	nComp := parent0.NComp()
	e.AddQuant(name, nComp, FIELD)

	sum := e.Quant(name)
	parents := make([]*Quant, len(args))
	for i := range parents {
		parents[i] = e.Quant(args[i])
	}
	e.Depends(name, args...)
	sum.updateSelf = &sumUpdater{sum, parents}
}

type sumUpdater struct {
	sum *Quant
	parents []*Quant
}

func (u *sumUpdater) Update() {
	// TODO: optimize for 0,1,2 or more parents
	sum := u.sum
	sum.array.Zero()	
	parents := u.parents
	for i := range parents{
		parent := parents[i]
		for c := range sum.Components{
			parComp := parent.array.Component[c]
			parMul := parent.multiplier[c]
			sumComp := sum.array.Component[c]
			gpu.Madd(sumComp.pointer, parComp.pointer, parMul)	
		}
	}
}
