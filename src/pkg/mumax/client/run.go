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
	"runtime"
)


// run the input files given on the command line
// todo: split in smaller functions
func run() {

	initOutputDir()

	// initialize the logger
	// it needs the output dir to log to
	// until now, everything went to stderr
	initLogger()
	Log(WELCOME)
	Debug("Go version:", runtime.Version())

	// make the FIFOs but do not yet try to open them
	makeFifos(outputDir())

	// run the sub-command (e.g. python) to interpret the script file
	// it will first hang while trying to open the FIFOs
	command := commandForFile(inputFile()) // e.g.: "python"
	proc := subprocess(command, flag.Args())
	Debug(command, "PID:", proc.Process.Pid)
	// start waiting for sub-command asynchronously and
	// use a channel to signal sub-command completion
	waiter := make(chan (int))
	go func() {
		msg, err := proc.Wait(0)
		if err != nil {
			panic(InputErr(err.String()))
		}
		waiter <- msg.ExitStatus() // send exit status to signal completion 
	}()
	// pipe sub-command output to the logger
	go logStream("["+command+":err]", proc.Stderr)
	go logStream("["+command+":out]", proc.Stdout)

	// open FIFOs for communication
	// there is a synchronization subtlety here:
	// opening the fifo's blocks until they have been
	// opened on the other side as well. So the subprocess
	// must be started first and must open the fifos in
	// the correct order (first OUT then IN).
	// this function hangs when the subprocess does not open the fifos.
	Debug("Opening FIFOs will block until", command, "opens the other end")
	var err os.Error
	//Debug("Opening", outfname)
	outfifo, err = os.OpenFile(outputDir()+"/"+OUTFIFO, os.O_WRONLY, 0666)
	CheckErr(err, ERR_BUG)
	//Debug("Opened ", outfname)
	//Debug("Opening", infname)
	infifo, err = os.OpenFile(outputDir()+"/"+INFIFO, os.O_RDONLY, 0666)
	CheckErr(err, ERR_IO)
	//Debug("Opened ", infname)

	// interpreter exports client methods
	c := new(Client)
	var ipc interpreter
	ipc.init(c)

	// interpreter executes commands from subprocess
	for line, eof := parseLine(infifo); !eof; line, eof = parseLine(infifo) {
		//Debug("call:", line)
		ret := ipc.call(line[0], line[1:])
		//Debug("return:", ret)
		switch len(ret) {
		default:
			panic(Bug("Method returned too many values"))
		case 0:
			fmt.Fprintln(outfifo)
		case 1:
			fmt.Fprintln(outfifo, ret[0])
		}
	}

	// wait for the sub-command to exit
	exitstat := <-waiter

	if exitstat != 0 {
		panic(InputErr(fmt.Sprint(command, " exited with status ", exitstat)))
	}
}


// make the output dir
func initOutputDir() {
	if *flag_rmoutput {
			err :=	syscommand("rm", []string{"-rf", outputDir()}) // ignore errors.
			if err != nil{
				Log("rm -rf", outputDir(), ":", err)
			}
	}
	errOut := os.Mkdir(outputDir(), 0777)
	CheckErr(errOut, ERR_IO)
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
	case ".class":
		return "java"
	case ".lua":
		return "lua"
	}
	panic(Bug("unreachable"))
	return ""
}


// pipes standard output/err of the command to the logger
// typically called in a separate goroutine
func logStream(prefix string, in io.Reader) {
	var bytes [BUFSIZE]byte
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

// IO buffer size
const BUFSIZE = 4096


// makes the FIFOs for inter-process communications
func makeFifos(outputDir string) {
	outfname := outputDir + "/" + OUTFIFO
	infname := outputDir + "/" + INFIFO
	cleanfiles = append(cleanfiles, infname, outfname)
	mkFifo(infname)
	mkFifo(outfname)

}


// Default FIFO filename for inter-process communication.
const (
	INFIFO  = "in.fifo"
	OUTFIFO = "out.fifo"
)
