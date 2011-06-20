//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

import (
	. "mumax/common"
	"flag"
	"runtime"
)


var (
//logFile     *string = flag.String("log", "mumax.log", "Specify the log file.")
//rpcType     *string = flag.String("rpc", "shell", "Specify RPC protocol: shell, (go, json)")
//interactive *bool   = flag.Bool("i", true, "interactive mode")
)

func Main() {
	flag.Parse()

	defer cleanup()
	initialize()
	run()
}


func run() {

}

func initialize() {
	InitLogger("mumax-engine.log")
	Log("mumax2 engine")
	Debug("Go version:", runtime.Version())
}

func cleanup() {
	Log("Finished.")
}
