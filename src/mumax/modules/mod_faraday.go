//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Module for Faraday's law.
// Author: Arne Vansteenkiste

//import (
//	. "mumax/engine"
//)
//
//// Register this module
//func init() {
//	RegisterModule("faraday", "Faraday's law", LoadFaraday)
//}
//
//// Load Faraday's law
//func LoadFaraday(e *Engine) {
//	LoadBField(e)
//	LoadEField(e)
//	e.AddTimeDerivative(e.Quant("B"))
//	maxwell.EnableFaraday(e.Quant("dB_dt"))
//	e.Depends("E", "dB_dt")
//}
