//  This file is part of muQueue, a job scheduler for MuMax.
//  Copyright 2012  Arne Vansteenkiste.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.

package main

// Implementation of "source" command

import ()

func init() {
	api["source"] = source
}

// read commands from file
func source(user *User, args []string) string {
	runFile(args[0])
	return "read " + args[0] + " (on server)"
}
