//  This file is part of muQueue, a job scheduler for MuMax.
//  Copyright 2012  Arne Vansteenkiste.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.

package main

// Job entry

import (
	"fmt"
	"os"
)

type Job struct {
	id      int      // unique identifier
	command []string // command and args to be executed
	user    *User    // job owner
	status  int      // queued, running, finished, failed
	node    *Node    // node this job is running on
	dev     []int    // devices this job is running on
	err     os.Error // error message, if any
	ndev int // number of requested devices
	nice int // priority
}

type JobStatus struct {
	*Job
	exitStatus os.Error
}

// job status
const (
	QUEUED = iota
	RUNNING
	FINISHED
	FAILED
)

var statusStr map[int]string = map[int]string{QUEUED: "que ", RUNNING: "run ", FINISHED: "done", FAILED: "fail"}

func NewJob(user *User, cmd []string) *Job {
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
		err = j.err.String()
	}
	return fmt.Sprint(printID(j.id), ":",
		"[nice", j.nice, "]",
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
