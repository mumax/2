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
	//"mumax/gpu"
)

// Register this module
func init() {
	RegisterModule("llb", "Landau-Lifshitz-Baryakhtar equation", LoadLLB)
}

// The torque quant contains the Landau-Lifshitz torque τ acting on the reduced magnetization m = M/Msat.
//	d m / d t =  τ  
// with unit
//	[τ] = 1/s
// Thus:
//	τ = gamma[ (m x h) - α m  x (m x h) ]
// with:
//	h = H / Msat
// To keep numbers from getting extremely large or small, 
// the multiplier is set to gamma, so the array stores τ/gamma
func LoadLLB(e *Engine) {

	LoadHField(e)
	LoadMagnetization(e)
	LoadFullMagnetization(e)
	
    e.LoadModule("baryakhtar")
    
    e.AddPDE1("mf", "torque")
}


