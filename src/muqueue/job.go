//  This file is part of muQueue, a job scheduler for MuMax.
//  Copyright 2012  Arne Vansteenkiste.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.

package main

// Job entry

import (
	"fmt"
)

type Job struct {
	id int
	file string
	user *User
	status int
}

// job status
const(
	QUEUED = iota
	RUNNING
	FINISHED
	FAILED
)

var statusStr map[int]string=map[int]string{QUEUED:"que ", RUNNING:"run ", FINISHED:"done", FAILED:"fail"}

func NewJob(user *User, cmd string) *Job {
	j := new(Job)
	j.file = cmd
	j.user = user
	j.id = nextID()
	return j
}

func (j *Job) String() string {
	return fmt.Sprint("[", printID(j.id), "]", "[", statusStr[j.status], "]","[", j.user, "] ", j.file)
}


var(
	lastID int
)

func nextID() int{
	lastID++
	return lastID
}

func printID(id int)string{
	return fmt.Sprintf("%08x", id)
}

