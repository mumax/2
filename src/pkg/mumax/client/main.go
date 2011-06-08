//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package client

import (
	. "mumax/common"
	"runtime"
)

const WELCOME = `MuMax 2.0.0.70 FD Multiphysics Client (C) Arne Vansteenkiste & Ben Van de Wiele, Ghent University.`

func Main() {
	initialize()
	defer cleanup()
}


func initialize() {
	InitLogger("mumax-client.log")
	Log(WELCOME)
	Debug("Go version:", runtime.Version())
}


func cleanup() {
	Log("Finished.")
}
