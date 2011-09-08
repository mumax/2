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
)


func (e *Engine) AddSumNode(name string, args ...string){
		parent0 := e.Quant(args[0])
		nComp := parent0.NComp()
		e.AddQuant(name, nComp, FIELD)
		sum := e.Quant(name)
		parents := make([]*Quant, len(args))
		for i:=range parents{
			parents[i] = e.Quant(args[i])
		}
		e.Depends(name, args...)
		sum.updateSelf = &sumUpdater{parents}
}

type sumUpdater struct {
	args []*Quant
}

func (u *sumUpdater) Update() {
	
}
