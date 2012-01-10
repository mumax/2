//  This file is part of muQueue, a job scheduler for MuMax.
//  Copyright 2012  Arne Vansteenkiste.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.

package main

// command to the scheduler
type Cmd struct {
	text     []string    // text-based command
	response chan string // chan to send answer and close connection
}
