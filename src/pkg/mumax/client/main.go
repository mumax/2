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


var ()

// Mumax2 main function
func Main() {
	defer func() {
		err := recover()
		if err != nil {
			crashreport(err)
		}
	}()

	initialize()
	run()
	cleanup()
}


func initialize() {
	InitLogger("mumax2.log")
	Log(WELCOME)
	Debug("Go version:", runtime.Version())
}


func run() {
	panic("hello panic")
}


func cleanup() {
	Log("Finished.")
}

func crashreport(err interface{}) {
	Log(err)
	Log("Crashed.")
}

const WELCOME = `MuMax 2.0.0.70 FD Multiphysics Client (C) Arne Vansteenkiste & Ben Van de Wiele, Ghent University.`
