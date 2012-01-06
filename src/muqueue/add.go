//  This file is part of muQueue, a job scheduler for MuMax.
//  Copyright 2012  Arne Vansteenkiste.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.

package main

// Implementation of "add" command

import (
	"fmt"
	. "mumax/common"
)

func init() {
	api["add"] = add
}

// adds a job
func add(user string, argz []string) string {
	if len(argz) == 0 {
		return "Nothing specified, nothing added.\nMaybe you wanted to say 'add command'?"
	}
	args, flags := parse(argz)
	job := NewJob(user, args)
	if nice, ok := flags["nice"]; ok {
		job.nice = Atoi(nice)
	}
	if gpus, ok := flags["gpu"]; ok {
		job.ndev = Atoi(gpus)
	}
	if user, ok := flags["user"]; ok {
		job.user = user
	}
	queue = append(queue, job)
	fillNodes()
	//log("added", job)
	return fmt.Sprint("added ", job)
}
