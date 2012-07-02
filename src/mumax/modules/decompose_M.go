//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

import (
    . "mumax/common"
	. "mumax/engine"
	"mumax/gpu"
)

type decomposeMUpdater struct {
    mf, msat0 *Quant
}

func (u *decomposeMUpdater) Update() {
    e := GetEngine()
    
    m := e.Quant("m")
    msat := e.Quant("Msat")
    
    mf := u.mf
    msat0 := u.msat0
    
    gpu.Limit(mf.Array(), msat0.Array(), float32(msat.Multiplier()[0]), float32(msat0.Multiplier()[0]))
	              
	gpu.Decompose(mf.Array(), 
	              m.Array(), 
	              msat.Array(), 
	              float32(msat.Multiplier()[0]))
	m.Invalidate()
	msat.Invalidate()
}
