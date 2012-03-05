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
var api map[string]func(*User, []string) string = make(map[string]func(*User, []string) string) // available commands

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

func fillNodes() {

	for _, node := range nodes {
		if node.Busy() {
			continue
		}

		ok := true
		for ok { // as long as we could start a job, try one more
			ok = false

			// find the job with highest priority for this node
			var bestjob *Job
			// find a job to start comparing priorities to
			for _, job := range queue {
				if job.status == QUEUED {
					bestjob = job
					break
				}
			}
			if bestjob == nil {
				break
			}

			for _, job := range queue {
				if job.status != QUEUED {
					continue
				}
				// first select on priority
				if job.nice < bestjob.nice {
					bestjob = job
					continue
				}
				// then prefer a user of this node's group
				if job.Group() == node.group && bestjob.Group() != node.group {
					bestjob = job
					continue
				}
				// TODO: then select on share
			}

			// now find a device on the node
			devices := freeDevice(bestjob, node)
			if devices != nil {
				dispatch(bestjob, node, devices)
				ok = true
			} else {
				log("job ", bestjob, " is stalling ", node)
			}
		}
	}
}

// finds a free node suited for the job.
// in case of multiple GPUs, they should be
// successive and aligned (to efficiently support GTX590s, e.g.)
func freeDevice(job *Job, n *Node) []int {
	if job == nil {
		return nil
	}
	ndev := job.ndev
	device := make([]int, ndev)
	for d := 0; d <= n.NDevice()-ndev; d++ {
		if d%ndev != 0 {
			continue
		}
		busy := false
		j := 0
		for i := d; i < d+ndev; i++ {
			device[j] = i
			if n.devBusy[i] {
				busy = true
			}
			j++
		}
		if !busy {
			return device
		}
	}
	return nil
}

// finds a free node suited for the job.
// in case of multiple GPUs, they should be
// successive and aligned (to efficiently support GTX590s, e.g.)
//func freeDevice(job *Job) (node *Node, device []int) {
//	if job == nil {
//		return
//	}
//	ndev := job.ndev
//	device = make([]int, ndev)
//	for _, n := range nodes {
//		for d := 0; d <= n.NDevice()-ndev; d++ {
//			if d%ndev != 0 {
//				continue
//			}
//			busy := false
//			j := 0
//			for i := d; i < d+ndev; i++ {
//				device[j] = i
//				if n.devBusy[i] {
//					busy = true
//				}
//				j++
//			}
//			if !busy {
//				node = n
//				return
//			}
//		}
//	}
//	return // nil,0 : nothing free
//}
