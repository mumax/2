//  This file is part of muQueue, a job scheduler for MuMax.
//  Copyright 2012  Arne Vansteenkiste.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.

package main

// Main program

import (
	"os"
)

func main() {
	if len(os.Args) == 1 {
		help()
		os.Exit(1)
	}
	args := os.Args[1:] // ommit program name

	if args[0] == "server" {
		serverMain()
		return
	} else {
		clientMain(args)
	}

}

func help() {
	err("Usage:", os.Args[0], "<command>", "[<args>]", "\n",
		"\nThe available commands are:")
}
