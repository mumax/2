//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// Author: Arne Vansteenkiste

import ()

// Register this module
func init() {
	RegisterModule("micromagnetism", "Standard micromagnetism", LoadMicromag)
}

func LoadMicromag(e *Engine) {
	e.LoadModule("magnetization")
	e.LoadModule("hfield")
	e.LoadModule("zeeman")
	e.LoadModule("aexchange")
	e.LoadModule("demagexch") // not yet enabled by default until well tested
	e.LoadModule("llg")
	e.LoadModule("regions")
}
