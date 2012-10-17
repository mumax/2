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
    "fmt"
)

// Register this module
func init() {
	RegisterModule("dissipative-function", "Dissipative function", LoadDF)
}


// There is a problem, since LLB torque is normalized by msat0 (equilibrium value), while LLG torque is normalized by msat
  
func LoadDF(e *Engine) {

    // make sure the effective field is in place
    LoadHField(e)
    
    Qmagn := e.AddNewQuant("Qmag", SCALAR, FIELD, Unit("J/(s*m3)"), "The heat flux density from magnetic subsystem to thermal bath")
    
    // THE UGLY PART STARTS HERE
    
    if e.HasQuant("mf") {
        e.Depends("Qmag", "torque", "H_eff", "msat0T0")
        Qmagn.SetUpdater(&DFUpdater{
		                  Qmagn: Qmagn,
		                  msat: e.Quant("msat0T0"),
		                  torque: e.Quant("torque"),
		                  Heff: e.Quant("H_eff")})
		
    } else if e.HasQuant("m") {
        e.Depends("Qmag", "torque", "H_eff", "msat")
        Qmagn.SetUpdater(&DFUpdater{
		                  Qmagn: Qmagn,
		                  msat: e.Quant("msat"),
		                  torque: e.Quant("torque"),
		                  Heff: e.Quant("H_eff")})
    } else {
        panic(InputErr(fmt.Sprint("There is no magnetic subsystem! Aborting.")))
    }
}

type DFUpdater struct {
	Qmagn *Quant
	msat  *Quant
	torque *Quant
	Heff *Quant
}

func (u *DFUpdater) Update() {

    // Account for msat multiplier, because it is a mask
	u.Qmagn.Multiplier()[0] = u.msat.Multiplier()[0]
	// Account for torque multiplier, because we usually put gamma there 
	u.Qmagn.Multiplier()[0] *= u.torque.Multiplier()[0]
	// Account for -mu0 
	u.Qmagn.Multiplier()[0] *= -0.5 * Mu0
	// Account for multiplier in H_eff
	u.Qmagn.Multiplier()[0] *= u.Heff.Multiplier()[0]
	// From now Qmag = dot(H_eff, torque)
	
	gpu.Dot(u.Qmagn.Array(),
	        u.Heff.Array(),
	        u.torque.Array())
	// Finally. do Qmag = Qmag * msat(r) to account spatial properties of msat
	if !u.msat.Array().IsNil() {
	    gpu.Mul(u.Qmagn.Array(),
	            u.Qmagn.Array(),
	            u.msat.Array())
	}
}
