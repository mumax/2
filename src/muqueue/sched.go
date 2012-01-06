//  This file is part of muQueue, a job scheduler for MuMax.
//  Copyright 2012  Arne Vansteenkiste.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.

package main

// Scheduler

import ()

// input from connections enters scheduler here
var (
	input  chan *Cmd      = make(chan *Cmd)      // takes input commands from user
	queue  []*Job         = make([]*Job, 0)      // stores queued and running jobs
	finish chan JobStatus = make(chan JobStatus) // returns finished jobs
	nodes  []*Node        = make([]*Node, 0)     // stores compute nodes
)

// available commands
var api map[string]func(string, []string) string = make(map[string]func(string, []string) string) // available commands

// run the scheduler
func runSched() {
	fillNodes()
	for {
		select {
		case cmd := <-input:
			cmd.response <- serveCommand(cmd.text) + "\n"
		case done := <-finish:
			undispatch(done.Job, done.exitStatus)
			fillNodes()
		}
	}
}

// returns the next job to be run
func nextJob() *Job {
	for _, job := range queue {
		if job.status == QUEUED {
			return job
		}
	}
	return nil
}

// remove a job from the list
func rmJob(job *Job, inList []*Job) (outList []*Job) {
	// find index
	i := 0
	for ; i < len(inList); i++ {
		if inList[i] == job {
			break
		}
	}
	if i == len(inList)-1 {
		outList = inList[:i]
	} else {
		outList = append(inList[:i], inList[i+1:]...)
	}
	return
}

// returns the first free node + device
func freeDevice() (node *Node, device int) {
	for _, n := range nodes {
		for d, busy := range n.devBusy {
			if !busy {
				node = n
				device = d
				return
			}
		}
	}
	return // nil,0 : nothing free
}
