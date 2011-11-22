//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// This file implements the average of a quantity.
// Author: Arne Vansteenkiste

import (
	"mumax/gpu"
)

type AverageUpdater struct {
	in, out *Quant
	reduce  gpu.Reductor
}

// TODO: what with masks?
func NewAverageUpdater(in, out *Quant) Updater {
	checkKind(in, FIELD)
	avg := new(AverageUpdater)
	avg.in = in
	avg.out = out
	avg.reduce.Init(1, GetEngine().GridSize())
	return avg
}

// TODO: what with masks?
func (this *AverageUpdater) Update() {
	for c := 0; c < this.in.nComp; c++ {
		sum := this.reduce.Sum(&(this.in.Array().Comp[c]))
		this.out.SetComponent(c, float64(sum)/float64(GetEngine().NCell()))
	}
}
