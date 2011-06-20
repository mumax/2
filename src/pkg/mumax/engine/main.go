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
	flag_port      *string
	flag_localAddr *string
	flag_logfile   *string
	flag_debug     *bool
	flag_silent    *bool
	flag_warn      *bool
)

func Main() {
	initFlags()

	defer cleanup()
	initialize()
	run()
}


// initializes flags for the engine
// note: this is not done inside the var() block to avoid
// clashing with the client flags.
func initFlags() {
	flag_port = flag.String("port", ":2527", "Set TCP incoming port")
	flag_localAddr = flag.String("local", "", "Local IP address to connect from")
	flag_logfile = flag.String("log", "", "Specify log file")
	flag_debug = flag.Bool("debug", true, "Show debug output")
	flag_silent = flag.Bool("silent", false, "Be silent")
	flag_warn = flag.Bool("warn", true, "Show warnings")
	flag.Parse()
}

func run() {
	listen()
}

func initialize() {
	initLogger()
	Log("mumax2 engine")
	Debug("Go version:", runtime.Version())
}


// initialize the logger
func initLogger() {
	var opts LogOption
	if !*flag_debug {
		opts |= LOG_NODEBUG
	}
	if *flag_silent {
		opts |= LOG_NOSTDOUT | LOG_NODEBUG | LOG_NOWARN
	}
	if !*flag_warn {
		opts |= LOG_NOWARN
	}

	logFile := *flag_logfile
	if logFile == "" {
		logFile = "mumax2-engine.log"
	}
	InitLogger(logFile, opts)
}

func cleanup() {
	Log("Finished.")
}
