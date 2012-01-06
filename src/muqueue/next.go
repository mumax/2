//  This file is part of muQueue, a job scheduler for MuMax.
//  Copyright 2012  Arne Vansteenkiste.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.

package main

// Implementation of "next" command

import ()

func init() {
	api["next"] = next
}

// reports next job to be dispatched
func next(user string, args []string) string {
	return nextJob().String()
}
