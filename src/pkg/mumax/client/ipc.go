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
	"strings"
	"io"
	"os"
)


// run the input files given on the command line
func runInputFile() {

	// run the sub-command (e.g. python) to interpret the script file
	command := commandForFile(inputFile()) // e.g.: "python"
	proc := subprocess(command, flag.Args())
	Debug(command, "PID:", proc.Process.Pid)

	// pipe sub-command output to the logger
	go logStream("["+command+":err]", proc.Stderr)
	go logStream("["+command+":out]", proc.Stdout)

	// make FIFOs for communication
	// there is a synchronization subtlety here:
	// opening the fifo's blocks until they have been
	// opened on the other side as well. So the subprocess
	// must be started first and must open the fifos in
	// the correct order (first OUT then IN).
	// this function hangs when the subprocess does not open the fifos.
	Debug("Opening FIFOs will block until", command, "opens the other end")
	makeFifos(outputDir())

	// wait for sub-command asynchronously and
	// use a channel to signal sub-command completion
	waiter := make(chan (int))
	go func() {
		msg, err := proc.Wait(0)
		if err != nil {
			panic(InputErr(err.String()))
		}
		waiter <- msg.ExitStatus() // send exit status to signal completion 
	}()

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

	// TODO: blocks until the other end is openend
	// to be moved until after subprocess is started
	var err os.Error
	//Debug("Opening", outfname)
	outfifo, err = os.OpenFile(outfname, os.O_WRONLY, 0666)
	CheckErr(err, ERR_BUG)
	//Debug("Opened ", outfname)

	//Debug("Opening", infname)
	infifo, err = os.OpenFile(infname, os.O_RDONLY, 0666)
	CheckErr(err, ERR_IO)
	//Debug("Opened ", infname)
}


// makes a fifo
// syscall.Mkfifo seems unavailable for the moment.
func mkFifo(fname string) {
	err := syscommand("mkfifo", []string{fname})
	if err != nil {
		panic(IOErr(fmt.Sprintf("mkfifo", fname, "returned", err)))
	}
}


func parseLine(in io.Reader) (words []string, eof bool) {
	str := ""
	var c byte
	c, eof = readChar(in)
	if eof {
		return
	}
	for c != '\n' {
		str += string(c)
		//Debug("str:", str)
		c, eof = readChar(in)
		if eof {
			return
		}
	}
	words = strings.Split(str, " ", -1)
	return
}

func readChar(in io.Reader) (char byte, eof bool) {
	var buffer [1]byte

	n := 0
	var err os.Error
	for n == 0 {
		n, err = in.Read(buffer[:])
		if err != nil {
			Debug(err)
			eof = true
			return
		}
	}
	char = buffer[0]
	return
}


// Default FIFO filename for inter-process communication.
const (
	INFIFO  = "in.fifo"
	OUTFIFO = "out.fifo"
)
