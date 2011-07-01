//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package frontend

// Utilities for executing external commands

import (
	. "mumax/common"
	"os"
	//"fmt"
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
	cmd := exec.Command(command, allargs...)
	cmd.Env = os.Environ()
	cmd.Dir = wd
	err3 := cmd.Start()

	//cmd, err3 := exec.Run(command, allargs, os.Environ(), wd, exec.PassThrough, exec.Pipe, exec.Pipe)
	output, err4 := cmd.CombinedOutput()
	CheckErr(err4, ERR_IO)
	Debug(string(output))
	if err3 != nil {
		panic(IOErr(err3.String()))
	}

	return cmd
}


// Runs the subprocess and waits for it to finish.
// The command is looked up in the PATH.
// Typically used for simple system commands: rm, mkfifo, cp, ... 
//func syscommand(command string, args []string) (err os.Error) {
//return (exec.Command(command, args...).Run())
//	cmd := subprocess(command, args)
//	werr := cmd.Wait()
//	err = werr
//	if err != nil{
//	if msg, ok := err.(*os.Waitmsg); ok  {
//		err = IOErr(fmt.Sprint(command, " exited with status ", msg.ExitStatus()))
//	}
//	}
//	return
//}
