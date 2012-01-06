//  This file is part of muQueue, a job scheduler for MuMax.
//  Copyright 2012  Arne Vansteenkiste.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.

package main

// Implementation of "rm" command

import (
	"fmt"
)

func init() {
	api["rm"] = rm
}

func rm(user string, args []string) (resp string) {
	if len(args) == 0 {
		resp = "rm needs an argument"
		return
	}

	count := 0
	denied := 0
	running := 0
	for i := 0; i < len(queue); i++ {
		job := queue[i]
		if match(job, args) { // TODO: kill if running
			if user == job.user {
				if job.status == RUNNING {
					running++ // do not remove running jobs
					continue
				}
				resp += "\n" + job.String()
				queue = rmJob(job, queue)
				count++
				i--
			} else {
				denied++
			}
		}
	}
	head := fmt.Sprint("removed ", count, " jobs from queue")

	resp = fmt.Sprint(head, resp)
	if denied != 0 {
		resp += fmt.Sprint("\ndid not remove ", denied, " jobs which you don't own")
	}
	if running != 0 {
		resp += fmt.Sprint("\ndid not remove ", running, " jobs which are running")
	}
	return
}
