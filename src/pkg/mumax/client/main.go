//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package client

import (
	. "mumax/common"
	cu "cuda/driver"
	"runtime"
	"runtime/debug"
	"fmt"
	"os"
	"flag"
)


var (
	flag_help      *bool   = flag.Bool("help", false, "Print help and exit")
	flag_outdir    *string = flag.String("o", "", "Override the standard output directory")
	flag_scriptcmd *string = flag.String("c", "", "Override the command for executing the source file. E.g.: python2.6")
	flag_debug     *bool   = flag.Bool("g", true, "Enable debug output")
	flag_silent    *bool   = flag.Bool("s", false, "Be silent")
	flag_warn      *bool   = flag.Bool("w", true, "Show warnings")
	flag_apigen    *bool   = flag.Bool("apigen", false, "Generate API files and exit")
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
	flag.Parse()
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
	InitLogger(LOGFILE, opts)
	Log(WELCOME)
	Debug("Go version:", runtime.Version())
}


func run() {
	if *flag_apigen {
		APIGen()
		return
	}
	if *flag_help {
		fmt.Fprintln(os.Stderr, "Usage:")
		flag.PrintDefaults()
		return
	}
	runInputFiles()
}


func cleanup() {
	exit(0)
}

func exit(status int) {
	Log("Exiting with error status", status, "(", ErrString[status], ")")
	os.Exit(status)
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
		Log("illegal input:", err, "\n", getCrashStack())
		status = ERR_INPUT
	case IOErr:
		Log("IO error:", err, "\n", getCrashStack())
		status = ERR_IO
	case cu.Result:
		Log("cuda error:", err, "\n", getCrashStack())
		status = ERR_CUDA
	}
	exit(status)
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


// Returns a stack trace for debugging an expected error (like IO).
// Only the stack element of the panic() call is shown.
//func getPanicCaller() string {
//	stack := debug.Stack()
//	// remove the first 8 lines, which are irrelevant
//	nlines := 0
//	start := 0
//	stop := 0
//	for i := range stack {
//		if stack[i] == byte('\n') {
//			nlines++
//		}
//		if nlines == 8 {
//			start = i + 1
//		}
//		if nlines == 10 {
//			stop = i + 1
//			break
//		}
//	}
//	return string(stack[start:stop])
//}


const (
	WELCOME  = `MuMax 2.0.0.70 FD Multiphysics Client (C) Arne Vansteenkiste & Ben Van de Wiele, Ghent University.`
	LOGFILE  = "mumax2.log"
	SENDMAIL = "\n-----\nIf you would like to have this issue fixed, please send \"" + LOGFILE + "\" to Arne.Vansteenkiste@UGent.be and/or Ben.VandeWiele@UGent.be\n-----\n"
)
