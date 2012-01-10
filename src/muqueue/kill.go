//  This file is part of muQueue, a job scheduler for MuMax.
//  Copyright 2012  Arne Vansteenkiste.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.

package main

// Implementation of "kill" command

import (
	. "mumax/common"
)

func init() {
	api["kill"] = kill
}

func kill(user *User, args []string) (resp string) {
	if len(args) == 0 {
		resp = "kill: need job id, e.g. 'kill 00000007'"
		return
	}
	for _, pid := range args {
		job := jobid(Atoi(pid))
		if job == nil {
			resp += "no such job: " + pid + "\n"
			continue
		}
		if job.user != user {
			resp += "kill job " + pid + ": permission denied\n"
		}
		if job.status != RUNNING {
			resp += "job not running: " + pid + "\n"
			continue
		}
		err := job.cmd.Process.Kill()
		if err == nil {
			resp += "killed job " + job.String() + "\n"
		} else {
			resp += "kill job " + pid + ": " + err.String() + "\n"
		}
	}
	return
}

// finds a job based on its id
func jobid(pid int) *Job {
	for _, job := range queue {
		if job.id == pid {
			return job
		}
	}
	return nil
}
