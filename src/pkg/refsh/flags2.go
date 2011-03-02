//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package refsh


// Parses command line arguments to refsh commands,
// according to the conventions of the Mercury Project.

// Author: Arne Vansteenkiste

import (
	"flag"
	. "strings"
)


// Parses the command line flags with the following layout:
// --command0="arg0, arg1" --command1="arg0" --command2 file1 file2 ...
// Returns an array of commands and args. command[i] has args args[i].
// files contains the CLI aruments that do not start with --
func ParseFlags2() (commands []string, args [][]string, files []string) {

	for i := 0; i < flag.NArg(); i++ {
		if HasPrefix(flag.Arg(i), "--") {

			cmd, arg := parseFlag2(flag.Arg(i))
			commands = append(commands, cmd)
			args = append(args, arg)

		} else {
			files = append(files, flag.Arg(i))
		}
	}

	return
}


// splits "--command="arg1, arg2" into "command", {arg1, arg2}
func parseFlag2(flag string) (command string, args []string) {
	assert(HasPrefix(flag, "--"))
	flag = flag[2:]
	split := Split(flag, "=", 2)
	command = split[0]
	if len(split) == 2 {
		args = Split(split[1], ",", -1)
		for i := range args {
			args[i] = TrimSpace(args[i])
		}
	}
	return
}


func assert(test bool) {
	if !test {
		panic("Assertion failed")
	}
}
