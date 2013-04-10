//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// This file implements the micromagnetism meta-module
// Author: Arne Vansteenkiste

import (
	. "mumax/engine"
)

// Register this module
func init() {
	RegisterModule("micromagnetism", "Basic micromagnetism module", LoadMicromag)
}

func LoadMicromag(e *Engine) {
	LoadHField(e)
	LoadMagnetization(e)
	e.LoadModule("demag")
	e.LoadModule("exchange6")
	e.LoadModule("llg")
	e.LoadModule("zeeman")
	if (e.HasSolver() == false) {
	   e.LoadModule("solver/rk12")
	   e.Quant("dt").SetScalar(1e-15)
	   e.Quant("mindt").SetScalar(1e-15)
	   e.Quant("m_maxerror").SetScalar(1. / 2000.)
	}
	e.LoadModule("maxtorque")
	/*torque := e.Quant("torque")

	maxtorque := e.AddNewQuant("maxtorque", SCALAR, VALUE, torque.Unit(), "Maximum |torque|")
	e.Depends("maxtorque", "torque")
	maxtorque.SetUpdater(NewMaxNormUpdater(torque, maxtorque))*/
}
