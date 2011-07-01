//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package frontend

import (
	. "mumax/common"
	cu "cuda/driver"
	"runtime"
	"runtime/debug"
	"runtime/pprof"
	"time"
	"fmt"
	"os"
	"flag"
)

// command-line flags (more in engine/main.go)
var (
	flag_outputdir *string = flag.String("out", "", "Specify output directory")
	flag_force     *bool   = flag.Bool("force", false, "Remove previous output directory if present")
	flag_logfile   *string = flag.String("log", "", "Specify log file")
	flag_command   *string = flag.String("command", "", "Override interpreter command")
	flag_debug     *bool   = flag.Bool("debug", true, "Show debug output")
	flag_cpuprof   *string = flag.String("cpuprof", "", "Write gopprof CPU profile to file")
	flag_memprof   *string = flag.String("memprof", "", "Write gopprof memory profile to file")
	flag_silent    *bool   = flag.Bool("silent", false, "Be silent")
	flag_warn      *bool   = flag.Bool("warn", true, "Show warnings")
	flag_help      *bool   = flag.Bool("help", false, "Print help and exit")
	flag_version   *bool   = flag.Bool("version", false, "Print version info and exit")
	flag_test      *bool   = flag.Bool("test", false, "Test CUDA and exit")
	flag_timeout   *string = flag.String("walltime", "", "Set a maximum run time. Units s,h,d are recognized.") // should be named walltime? timeout=only for connection?
)


// Mumax2 main function
func Main() {
	// first test for flags that do not actually run a simulation
	flag.Parse()

	defer func() {
		cleanup()        // make sure we always clean up, no matter what
		err := recover() // if anything goes wrong, produce a nice crash report
		if err != nil {
			crashreport(err)
		}
	}()

	if *flag_cpuprof != "" {
		f, err := os.Create(*flag_cpuprof)
		if err != nil {
			Log(err)
		}
		Log("Writing CPU profile to", *flag_cpuprof)
		pprof.StartCPUProfile(f)
		// will be flushed on cleanup
	}

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

	initTimeout()

	// else...
	clientMain()
}


// return the input file. "" means none
func inputFile() string {
	// check if there is just one input file given on the command line
	if flag.NArg() == 0 {
		//panic(InputErr("no input files"))
		return ""
	}
	if flag.NArg() > 1 {
		panic(InputErr(fmt.Sprint("need exactly 1 input file, but", flag.NArg(), "given:", flag.Args())))
	}
	return flag.Arg(0)
}


func cleanup() {
	Debug("cleanup")

	// write memory profile
	if *flag_memprof != "" {
		f, err := os.Create(*flag_memprof)
		if err != nil {
			Log(err)
		}
		Log("Writing memory profile to", *flag_memprof)
		pprof.WriteHeapProfile(f)
		f.Close()
	}

	// write cpu profile
	if *flag_cpuprof != "" {
		Log("Flushing CPU profile", *flag_cpuprof)
		pprof.StopCPUProfile()
	}

	// remove neccesary files
	//for i := range cleanfiles {
	//	Debug("rm", cleanfiles[i])
	//	err := os.Remove(cleanfiles[i])
	//	if err != nil {
	//		Debug(err)
	//	} // ignore errors, there's nothing we can do about it during cleanup
	//}

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


// sets up a timeout that will kill mumax when it runs too long
func initTimeout() {
	timeout := *flag_timeout
	t := 0.
	if timeout != "" {
		switch timeout[len(timeout)-1] {
		default:
			t = Atof64(timeout)
		case 's':
			t = Atof64(timeout[:len(timeout)-1])
		case 'h':
			t = 3600 * Atof64(timeout[:len(timeout)-1])
		case 'd':
			t = 24 * 3600 * Atof64(timeout[:len(timeout)-1])
		}
	}
	if t != 0 {
		Log("Timeout: ", t, "s")
		go func() {
			time.Sleep(int64(1e9 * t))
			Log("Timeout reached:", timeout)
			cleanup()
			os.Exit(ERR_IO)
		}()
	}
}

const (
	WELCOME  = `MuMax 2.0.0.70 FD Multiphysics Client (C) Arne Vansteenkiste & Ben Van de Wiele, Ghent University.`
	LOGFILE  = "mumax2.log"
	SENDMAIL = "\n-----\nIf you would like to have this issue fixed, please send \"" + LOGFILE + "\" to Arne.Vansteenkiste@UGent.be and/or Ben.VandeWiele@UGent.be\n-----\n"
)
