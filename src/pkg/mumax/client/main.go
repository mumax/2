//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package client

import (
	. "mumax/common"
	"mumax/engine"
	cu "cuda/driver"
	"runtime"
	"runtime/debug"
	"fmt"
	"os"
	"flag"
)




// client global variables
var (
	cleanfiles []string // list of files to be deleted upon program exit
)

// Mumax2 main function
func Main() {
	// first test for flags that do not actually run a simulation
	flag.Parse()
	if *engine.Flag_help {
		fmt.Fprintln(os.Stderr, "Usage:")
		flag.PrintDefaults()
		return
	}
	if *engine.Flag_version {
		fmt.Println(WELCOME)
		fmt.Println("Go", runtime.Version())
		return
	}
	if *engine.Flag_apigen {
		APIGen()
		return
	}

	// actual run
	defer func() {
		cleanup()        // make sure we always clean up, no matter what
		err := recover() // if anything goes wrong, produce a nice crash report
		if err != nil {
			crashreport(err)
		}
	}()

	if *engine.Flag_test{
		cu.Init()
		return
	}
	if *engine.Flag_engine{
			engine.Run()
			return
	}
	// else...
	run()
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
	if *engine.Flag_outputdir != "" {
		return *engine.flag_Outputdir
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
