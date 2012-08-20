//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// File provides various temperature quantities
// Author: Mykola Dvornik

import (
	. "mumax/engine"
	. "mumax/common"
	"mumax/gpu"
)

// Load the temperatures
func LoadGes(e *Engine) {
    //LoadHField(e)
	//LoadFullMagnetization(e)
	
	if e.HasQuant("h_eff") && e.HasQuant("m") && !e.HasQuant("Ges") {
	    Debug("3TM is initalized with magnetization dynamics support.")
		Ges := e.AddNewQuant("Ges", SCALAR, FIELD, Unit("W/(K*m^3)"), "The electron-spin coupling coefficient")
		g_es := e.AddNewQuant("g_es", SCALAR, MASK, Unit(""), "The dimensionless electron-spin coupling coefficient (Giblert constant)")
		e.Depends("Ges", "g_es", "m", "msat", "h_eff", "mf")
	    Ges.SetUpdater(&GesUpdater{g_es: g_es, Ges: Ges, m: e.Quant("m"), msat: e.Quant("msat"), h_eff: e.Quant("h_eff")})
	} else {
	    Debug("3TM is initalized without magnetization dynamics support")
	    e.AddNewQuant("Ges", SCALAR, MASK, Unit("W/(K*m^3)"), "The electron-spin coupling coefficient")
	}
	
}

type GesUpdater struct {
	g_es, Ges, m, msat, h_eff *Quant
}

func (u *GesUpdater) Update() {
    g_es := u.g_es
    msat := u.msat
    m := u.m
    h_eff := u.h_eff
    Ges := u.Ges
    
    mult := -2.0 * Mu0 * Kb * g_es.Multiplier()[0] * msat.Multiplier()[0] / H_bar // minus is due to different conventions for relaxation constant
    Ges.Multiplier()[0] = mult
    
    
    // Do dot(m,H) => Ges
    gpu.Dot(Ges.Array(), m.Array(), h_eff.Array())
    // Do msat * Ges => Ges
    gpu.Mul(Ges.Array(), Ges.Array(), msat.Array())
    
    // If g_es is an array then do Ges * g_es => Ges pointwise
    if !g_es.Array().IsNil() {
        gpu.Mul(Ges.Array(), Ges.Array(), g_es.Array())
    }
    
}
