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
	"path"
	"io"
	"os"
)

// run the input files given on the command line
func runInputFiles() {
	// check if there is just one input file given on the command line
	if flag.NArg() == 0 {
		Log("No input files")
		return
	}
	if flag.NArg() > 1 {
		Log("Need exactly 1 input file, but", flag.NArg(), "given:", flag.Args())
	}

	file := flag.Arg(0)
	command := commandForFile(file)

	proc := subprocess(command, flag.Args())
	Debug(command, "PID:", proc.Process.Pid)

	go logStream("[" + command + "]", proc.Stderr)
	go logStream("[" + command + "]", proc.Stdout)

	msg, err := proc.Wait(0)
	if err != nil {
		panic(InputErr(err.String()))
	}

	stat := msg.ExitStatus()
	Debug(command, "exited with status", stat)

	if stat != 0 {
		exit(ERR_INPUT)
	}
}


// given a file name (e.g. file.py)
// this returns a command to run the file (e.g. python)
func commandForFile(file string) string {
	if *scriptcmd != "" {
		return *scriptcmd
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
func logStream(prefix string, in io.Reader){
	var bytes [512]byte
	buf := bytes[:]
	var err os.Error = nil
	n:=0
	for err == nil{
		n,err= in.Read(buf)
		if n != 0{Log(prefix, string(buf))}// TODO: no printLN
	}
}
