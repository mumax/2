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
	"runtime/debug"
	"fmt"
	"os"
	"flag"
)


var (
	help   *bool = flag.Bool("help", false, "Print help and exit")
	outdir *string = flag.String("o", "", "Override the standard output directory")
	apigen *bool = flag.Bool("apigen", false, "Generate API files and exit")
)

// Mumax2 main function
func Main() {
	// if anything goes wrong, produce a nice crash report
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
	InitLogger(LOGFILE)
	Log(WELCOME)
	Debug("Go version:", runtime.Version())
	flag.Parse()
}


func run() {
	if *apigen {
		APIGen()
		return
	}
	if *help {
		fmt.Fprintln(os.Stderr, "Usage:")
		flag.PrintDefaults()
		return
	}
	
	runInputFiles()	
}


func cleanup() {
	Log("Finished")
}

func crashreport(err interface{}) {
	stack := GetCrashStack()
	Log("panic:", err, "\n", stack)
	Log("If you think this is a bug, please send the log file \"" + LOGFILE + "\" to Arne.Vansteenkiste@UGent.be and/or Ben.VandeWiele@UGent.be")
	stat := 1
	Log("Exiting with error status", stat)
	os.Exit(stat)
}

// Returns a stack trace for debugging a crash.
// The first irrelevant lines are discarded
// (they trace to this function), so the trace
// starts with the relevant panic() call.
func GetCrashStack() string {
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
	WELCOME = `MuMax 2.0.0.70 FD Multiphysics Client (C) Arne Vansteenkiste & Ben Van de Wiele, Ghent University.`
	LOGFILE = "mumax2.log"
)
