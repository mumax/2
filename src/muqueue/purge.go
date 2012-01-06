//  This file is part of muQueue, a job scheduler for MuMax.
//  Copyright 2012  Arne Vansteenkiste.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.

package main

// Implementation of "purge" command

import (
	"fmt"
	"runtime"
)

func init() {
	api["purge"] = purge
}

// purge failed or done jobs from list
func purge(user string, args []string) string {

	count := 0
	for i := 0; i < len(queue); i++ {
		job := queue[i]
		if job.Finished() && match(job, args) {
			queue = rmJob(job, queue)
			count++
			i--
		}
	}
	runtime.GC()
	return fmt.Sprint("purged ", count, " finished jobs from list")
}
