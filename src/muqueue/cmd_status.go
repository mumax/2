//  This file is part of muQueue, a job scheduler for MuMax.
//  Copyright 2012  Arne Vansteenkiste.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.

package main

// Implementation of "status" command

import (
	"fmt"
)

func init() {
	api["status"] = status
}

// reports the queue status
func status(user *User, args []string) string {
	status := fmt.Sprint(len(queue), " Jobs queued\n")
	for _, job := range queue {
		status += fmt.Sprint(job, "\n")
	}
	return status
}
