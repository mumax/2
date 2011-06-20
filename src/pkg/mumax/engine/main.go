//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

import (
	//. "mumax/common"
	"flag"
	//"runtime"
)


// command-line flags
var (
	Flag_engine  *bool   = flag.Bool("engine", false, "Run engine on incoming port")
	Flag_port    *string = flag.String("port", ":2527", "Set TCP listen port for engine")
	Flag_net     *string = flag.String("net", "tcp", "Set network: tcp[4,6], udp[4,6], unix[gram]")
	Flag_outputdir *string = flag.String("out", "", "Specify output directory")
	Flag_logfile *string = flag.String("log", "", "Specify log file")
	Flag_debug   *bool   = flag.Bool("debug", true, "Show debug output")
	Flag_silent  *bool   = flag.Bool("silent", false, "Be silent")
	Flag_warn    *bool   = flag.Bool("warn", true, "Show warnings")
	Flag_help    *bool   = flag.Bool("help", false, "Print help and exit")
	Flag_version *bool   = flag.Bool("version", false, "Print version info and exit")
	Flag_test    *bool   = flag.Bool("test", false, "Test CUDA and exit")
	Flag_apigen  *bool   = flag.Bool("apigen", false, "Generate API and exit (internal use)")
)

func Run() {
	listen()
}
