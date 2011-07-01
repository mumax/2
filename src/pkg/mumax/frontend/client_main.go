//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package frontend

// This file implements the mumax2's main routines for client mode.

import (
	. "mumax/common"
	"path"
	"os"
	"runtime"
)

// run the input files given on the command line
func clientMain() {
	infile := inputFile()
	outdir := outputDir(infile)
	initOutputDir(outdir)
	initLogger(outdir)
	Log(WELCOME)
	Debug("Go", runtime.Version())
	command := *flag_command

	var client Client
	client.Init(infile, outdir, command)
	client.Run()

	//	if *flag_test {
	//		var result string
	//		err := server.Call("engine.Test", &VoidArgs{}, &result)
	//		CheckErr(err, ERR_BUG)
	//		Log(result)
	//		return
	//	}

}


// return the output directory
func outputDir(inputFile string) string {
	if *flag_outputdir != "" {
		return *flag_outputdir
	}
	return inputFile + ".out"
}


// make the output dir
func initOutputDir(outputDir string) {
	if *flag_force {
		err := syscommand("rm", []string{"-rf", outputDir})
		if err != nil {
			Log("rm -rf", outputDir, ":", err)
		}
	}
	errOut := os.Mkdir(outputDir, 0777)
	CheckErr(errOut, ERR_IO)

}


// initialize the logger
func initLogger(outputDir string) {
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
		logFile = outputDir + "/mumax2.log"
	}
	InitLogger(logFile, opts)
}


// given a file name (e.g. file.py)
// this returns a command to run the file (e.g. python file.py, java File)
func commandForFile(file string) (command string, args []string) {
	if *flag_command != "" {
		return *flag_command, []string{file}
	}
	if file == "" {
		panic(IOErr("no input file"))
	}
	switch path.Ext(file) {
	default:
		panic(InputErr("Cannot handle files with extension " + path.Ext(file)))
	case ".py":
		return "python", []string{file}
	case ".java":
		return GetExecDir() + "javaint", []string{file}
	case ".class":
		return "java", []string{ReplaceExt(file, "")}
	case ".lua":
		return "lua", []string{file}
	}
	panic(Bug("unreachable"))
	return "", nil
}


const (
	INFIFO        = "in.fifo"   // FIFO filename for mumax->subprocess text-based function calls.
	OUTFIFO       = "out.fifo"  // FIFO filename for mumax<-subprocess text-based function calls.
	HANDSHAKEFILE = "handshake" // Presence of this file indicates subprocess initialization OK.
)
