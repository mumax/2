//  This file is part of muQueue, a job scheduler for MuMax.
//  Copyright 2012  Arne Vansteenkiste.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.

package main

// Implementation of "add" command

import (
	. "mumax/common"
	"fmt"
	"exec"
	"path"
)

func init() {
	api["add"] = add
}

// adds a job
func add(user *User, argz []string) string {
	if len(argz) == 0 {
		return "Nothing specified, nothing added.\nMaybe you wanted to say 'add command'?"
	}
	args, flags := parse(argz, "nice", "gpus", "user")
	job := NewJob(user, args)

	// replace command by full path
	shortCommand := args[0]
	full, err := exec.LookPath(shortCommand)
	if err == nil {
		args[0] = path.Clean(full)
	} else {
		args[0] = shortCommand
	}

	if nice, ok := flags["nice"]; ok {
		job.nice = Atoi(nice)
	}
	if gpus, ok := flags["gpus"]; ok {
		job.ndev = Atoi(gpus)
	}
	if username, ok := flags["user"]; ok {
		job.user = GetUser(username)
	}
	queue = append(queue, job)
	fillNodes()
	//log("added", job)
	return fmt.Sprint(job)
}
