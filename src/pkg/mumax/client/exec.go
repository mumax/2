//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package client

// Utilities for executing external commands

import (
	. "mumax/common"
	"os"
	"exec"
)

// Wrapper for exec.Run.
// Automatically looks up the executable in the PATH
// Uses the current working directory and environment.
func subprocess(command string, args []string) *exec.Cmd {
	command, err := exec.LookPath(command)
	if err != nil {
		panic(InputErr(err.String()))
	}
	allargs := []string{command} // argument 1, not argument 0 is the first real argument, argument 0 is the program name
	allargs = append(allargs, args...)

	wd, errwd := os.Getwd()
	if errwd != nil {
		panic(IOErr(errwd.String()))
	}

	Debug("exec", allargs)
	cmd, err3 := exec.Run(command, allargs, os.Environ(), wd, exec.PassThrough, exec.Pipe, exec.Pipe)
	if err3 != nil {
		panic(IOErr(err3.String()))
	}
	return cmd
}


// Runs the subprocess and waits for it to finish.
// The command is looked up in the PATH.
// Output is passed through to stdout/stderr.
// Typically used for simple system commands: rm, mkfifo, cp, ... 
//func syscommand(command string, args []string) (err os.Error) {
//	command, err = exec.LookPath(command)
//	if err != nil {
//		return
//	}
//	cmd, err2 := subprocess(command, args, exec.DevNull, exec.PassThrough, exec.PassThrough)
//	err = err2
//	if err != nil {
//		return
//	}
//	_, err = cmd.Wait(0)
//	return
//}
