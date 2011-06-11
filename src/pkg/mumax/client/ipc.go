//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package client

// This file implements Inter-Process-Communication
// between mumax and a scripting language.

import (
	. "mumax/common"
	"flag"
	"fmt"
	"path"
	"io"
	"os"
)


// run the input files given on the command line
func runInputFile() {
	// make FIFOs for communciation
	makeFifos(outputDir)

	// run the sub-command (e.g. python) to interpret the script file
	command := commandForFile(inputFile)
	proc := subprocess(command, flag.Args())
	Debug(command, "PID:", proc.Process.Pid)

	// pipe sub-command output to the logger
	go logStream("["+command+":err]", proc.Stderr)
	go logStream("["+command+":out]", proc.Stdout)

	// wait for sub-command asynchronously and
	// use a channel to signal sub-command completion
	waiter := make(chan (int))
	exitstat := 0
	go func() {
		msg, err := proc.Wait(0)
		if err != nil {
			panic(InputErr(err.String()))
		}
		exitstat = msg.ExitStatus()
		Debug(command, "exited with status", exitstat)
		waiter <- 1 // send dummy value to signal completion
	}()

	// wait for the sub-command to exit
	<-waiter

	if exitstat != 0 {
		Exit(ERR_INPUT)
	}
}


// given a file name (e.g. file.py)
// this returns a command to run the file (e.g. python)
func commandForFile(file string) string {
	if *flag_scriptcmd != "" {
		return *flag_scriptcmd
	}
	switch path.Ext(file) {
	default:
		panic(InputErr("Cannot handle files with extension " + path.Ext(file)))
	case ".py":
		return "python"
	}
	panic(Bug("unreachable"))
	return ""
}


// pipes standard output/err of the command to the logger
// typically called in a separate goroutine
func logStream(prefix string, in io.Reader) {
	var bytes [512]byte
	buf := bytes[:]
	var err os.Error = nil
	n := 0
	for err == nil {
		n, err = in.Read(buf)
		if n != 0 {
			Log(prefix, string(buf[:n]))
		} // TODO: no printLN
	}
}


// makes the FIFOs for inter-process communications
func makeFifos(outputDir string) {
	mkFifo(outputDir + "/" + OUTFIFO)
	mkFifo(outputDir + "/" + INFIFO)
}


// makes a fifo
// syscall.Mkfifo seems unavailable for the moment.
func mkFifo(fname string) {
	err := syscommand("mkfifo", []string{fname})
	if err != nil {
		panic(IOErr(fmt.Sprintf("mkfifo", fname, "returned", err)))
	}
}

// Default FIFO filename for inter-process communication.
const (
	INFIFO  = "in.fifo"
	OUTFIFO = "out.fifo"
)
