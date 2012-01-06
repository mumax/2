//  This file is part of muQueue, a job scheduler for MuMax.
//  Copyright 2012  Arne Vansteenkiste.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.

package main

// Job entry

import (
	. "mumax/common"
	"fmt"
	"os"
)

type Job struct {
	id      int      // unique identifier
	command []string // command and args to be executed
	user    string   // job owner
	status  int      // queued, running, finished, failed
	node    *Node    // node this job is running on
	dev     []int    // devices this job is running on
	err     os.Error // error message, if any
	ndev    int      // number of requested devices
	nice    int      // priority
}

type JobStatus struct {
	*Job
	exitStatus os.Error
}

// job status
const (
	QUEUED = iota
	RUNNING
	DONE
	FAILED
)

var statusStr map[int]string = map[int]string{QUEUED: "que ",
	RUNNING: YELLOW + BOLD + "run " + RESET,
	DONE:    GREEN + "done" + RESET,
	FAILED:  RED + BOLD + "FAIL" + RESET}

func NewJob(user string, cmd []string) *Job {
	j := new(Job)
	j.command = cmd
	j.user = user
	j.id = nextID()
	j.ndev = 1 // default, may be overridden
	j.nice = 2 // default, may be overridden
	return j
}

func (j *Job) String() string {
	if j == nil {
		return "<no job>"
	}
	err := ""
	if j.err != nil {
		err = RED+BOLD+j.err.String()+RESET
	}
	return fmt.Sprint(printID(j.id),
		" ", statusStr[j.status],
		" ", j.user,
		" nice", j.nice,
		" ", j.ndev, "gpu ",
		j.command, " ", err)
		//append([]string{path.Base(j.command[0])}, j.command[1:]...), " ", err)
}

func (j *Job) LongString() string {
	if j == nil {
		return "<no job>"
	}
	err := ""
	if j.err != nil {
		err = j.err.String()
	}
	return fmt.Sprint(printID(j.id), " ",
		"nice", j.nice, "]",
		"[", j.ndev, "GPU]",
		"[", statusStr[j.status], "]",
		"[", j.user, "] ",
		j.command, " ", err)
}

var (
	lastID int
)

func nextID() int {
	lastID++
	return lastID
}

func printID(id int) string {
	return fmt.Sprintf("%08x", id)
}

// done or failed
func (job *Job) Finished() bool {
	return job.status == DONE || job.status == FAILED
}
