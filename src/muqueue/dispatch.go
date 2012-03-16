//  This file is part of muQueue, a job scheduler for MuMax.
//  Copyright 2012  Arne Vansteenkiste.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.

package main

import (
	"fmt"
	. "mumax/common"
	"os/exec"
)

// Starts the job
func dispatch(job *Job, node *Node, dev []int) string {

	// set job status
	job.status = RUNNING
	job.node = node
	job.dev = dev

	// set node status
	for _, d := range dev {
		node.devBusy[d] = true
	}

	// start command
	devs := fmt.Sprint(dev[0])
	for i := 1; i < len(dev); i++ {
		Assert(!node.devBusy[dev[i]])
		devs += fmt.Sprint(",", dev[i])
	}
	// insert -gpu=...
	job.command = append(job.command[:1], append([]string{"-gpu=" + devs}, job.command[1:]...)...)
	ssh := node.loginCmd
	job.command = append(ssh, job.command...)

	job.cmd = exec.Command(job.command[0], job.command[1:]...)
	go func() {
		log(job.command)
		err := job.cmd.Run()
		out, _ := job.cmd.CombinedOutput()
		log(string(out))
		finish <- JobStatus{job, err}
	}()

	// report to user
	return fmt.Sprint("dispatched ", job, " to ", node.hostname, ":", devs)
}

// Reports the job done
func undispatch(job *Job, exitStatus error) {

	// set job status
	job.err = exitStatus
	job.cmd = nil // allow GC
	if exitStatus == nil {
		job.status = DONE
	} else {
		job.status = FAILED
	}

	// set node status
	for _, d := range job.dev {
		job.node.devBusy[d] = false
	}
	log(job)
}

// Starts the next job in the queue
//func dispatchNext() string {
//	job := nextJob()
//	if job == nil {
//		return "No jobs in queue"
//	}
//	node, dev := freeDevice(job)
//	if node == nil {
//		return "No free device"
//	}
//	return dispatch(job, node, dev)
//}

//func init() {
//	api["dispatch"] = dispatchManual
//}

// Manual dispatch
//func dispatchManual(user *User, args []string) string {
//	if len(args) == 0 {
//		return dispatchNext()
//	}
//
//	resp := ""
//	return resp
//}
