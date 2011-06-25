//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

import (
	. "mumax/common"
	cu "cuda/driver"
	"runtime"
	"runtime/debug"
	"runtime/pprof"
	"fmt"
	"os"
	"flag"
)


// command-line flags (more in engine/main.go)
var (
	flag_engine     *bool   = flag.Bool("listen", false, "Run engine on incoming port")
	flag_engineAddr *string = flag.String("connect", "", "Connect to engine server")
	flag_outputdir  *string = flag.String("out", "", "Specify output directory")
	flag_force      *bool   = flag.Bool("force", false, "Remove previous output directory if present")
	flag_logfile    *string = flag.String("log", "", "Specify log file")
	flag_scriptcmd  *string = flag.String("command", "", "Override interpreter command")
	flag_debug      *bool   = flag.Bool("debug", true, "Show debug output")
	flag_cpuprof    *string = flag.String("cpuprof", "", "Write CPU profile to file")
	flag_memprof    *string = flag.String("memprof", "", "Write memory profile to file")
	flag_silent     *bool   = flag.Bool("silent", false, "Be silent")
	flag_warn       *bool   = flag.Bool("warn", true, "Show warnings")
	flag_help       *bool   = flag.Bool("help", false, "Print help and exit")
	flag_version    *bool   = flag.Bool("version", false, "Print version info and exit")
	flag_test       *bool   = flag.Bool("test", false, "Test CUDA and exit")
	flag_apigen     *bool   = flag.Bool("apigen", false, "Generate API and exit (internal use)")
	flag_port       *string = flag.String("port", ":2527", "Set TCP listen port for engine")
	flag_net        *string = flag.String("net", "tcp", "Set network: tcp[4,6], udp[4,6], unix[gram]")
)


// client global variables
var (
	cleanfiles []string // list of files to be deleted upon program exit
)

// Mumax2 main function
func Main() {
	// first test for flags that do not actually run a simulation
	flag.Parse()
	if *flag_help {
		fmt.Fprintln(os.Stderr, "Usage:")
		flag.PrintDefaults()
		return
	}
	if *flag_version {
		fmt.Println(WELCOME)
		fmt.Println("Go", runtime.Version())
		return
	}
	if *flag_apigen {
		APIGen()
		return
	}
	if *flag_cpuprof {
		f, err := os.Create(*flag_cpuprof)
		CheckErr(err, ERR_IO)
		Log("Writing CPU profile to", *flag_cpuprof)
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	defer func() {
		cleanup()        // make sure we always clean up, no matter what
		err := recover() // if anything goes wrong, produce a nice crash report
		if err != nil {
			crashreport(err)
		}
	}()

	if *flag_test {
		cu.Init()
		return
	}
	if *flag_engine {
		listen()
		return
	}
	// else...
	run()

	// memory profile is single-shot, run at the end of program
	if *flag_memprof {
		f, err := os.Create(*flag_memprof)
		CheckErr(err, ERR_IO)
		Log("Writing memory profile to", *flag_memprof)
		pprof.WriteHeapProfile(f)
		f.Close()
	}
}


// return the input file
func inputFile() string {
	// check if there is just one input file given on the command line
	if flag.NArg() == 0 {
		panic(InputErr("no input files"))
	}
	if flag.NArg() > 1 {
		panic(InputErr(fmt.Sprint("need exactly 1 input file, but", flag.NArg(), "given:", flag.Args())))
	}
	return flag.Arg(0)
}


// return the output directory
func outputDir() string {
	if *flag_outputdir != "" {
		return *flag_outputdir
	}
	return inputFile() + ".out"
}


func cleanup() {
	Debug("cleanup")

	// remove neccesary files
	for i := range cleanfiles {
		Debug("rm", cleanfiles[i])
		err := os.Remove(cleanfiles[i])
		if err != nil {
			Debug(err)
		} // ignore errors, there's nothing we can do about it during cleanup
	}

	// kill subprocess
}


func crashreport(err interface{}) {
	status := 0
	switch err.(type) {
	default:
		Log("panic:", err, "\n", getCrashStack())
		Log(SENDMAIL)
		status = ERR_PANIC
	case Bug:
		Log("bug:", err, "\n", getCrashStack())
		Log(SENDMAIL)
		status = ERR_BUG
	case InputErr:
		Log("illegal input:", err, "\n")
		if *flag_debug {
			Log(getCrashStack())
		}
		status = ERR_INPUT
	case IOErr:
		Log("IO error:", err, "\n")
		if *flag_debug {
			Log(getCrashStack())
		}
		status = ERR_IO
	case cu.Result:
		Log("cuda error:", err, "\n", getCrashStack())
		if *flag_debug {
			Log(getCrashStack())
		}
		status = ERR_CUDA
	}
	Log("Exiting with status", status, ErrString[status])
	os.Exit(status)
}

// Returns a stack trace for debugging a crash.
// The first irrelevant lines are discarded
// (they trace to this function), so the trace
// starts with the relevant panic() call.
func getCrashStack() string {
	stack := debug.Stack()
	// remove the first 8 lines, which are irrelevant
	nlines := 0
	start := 0
	for i := range stack {
		if stack[i] == byte('\n') {
			nlines++
		}
		if nlines == 8 {
			start = i + 1
			break
		}
	}
	return string(stack[start:])
}


const (
	WELCOME  = `MuMax 2.0.0.70 FD Multiphysics Client (C) Arne Vansteenkiste & Ben Van de Wiele, Ghent University.`
	LOGFILE  = "mumax2.log"
	SENDMAIL = "\n-----\nIf you would like to have this issue fixed, please send \"" + LOGFILE + "\" to Arne.Vansteenkiste@UGent.be and/or Ben.VandeWiele@UGent.be\n-----\n"
)
