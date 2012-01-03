//  This file is part of muQueue, a job scheduler for MuMax.
//  Copyright 2012  Arne Vansteenkiste.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.

package main

// Scheduler

import (
	"fmt"
)

// input from connections enters scheduler here
var (
	input chan *Cmd = make(chan *Cmd)// takes input commands from user
	donejob chan *Job = make(chan *Job)// finished jobs are returned here
	queue  []*Job = make([]*Job,0)
	pending []*Job = make([]*Job,0)
	nodes []*Node = make([]*Node,0)
)


// command to the scheduler
type Cmd struct {
	text     string      // text-based command
	response chan string // chan to send answer and close connection
}

// initialize the scheduler
func initSched() {
	// TODO: read from config
	nodes = append(nodes, NewNode("localhost", 1))
}

// run the scheduler
func runSched() {

	fillNodes()

	for {
		select {
		case cmd := <-input:
			cmd.response <- serveCommand(cmd.text)
		case done := <- donejob:
			rmJob(done, pending)	
			fillNodes()	
		}
	}
}

// processes a command issued by user
func serveCommand(line string) (response string) {
	log("got command ", line)
	queue = append(queue, NewJob(line))
	return fmt.Sprint("added job ", len(queue), ": ", line, "\n")
}


func dispatchJob(job *Job){

}


// returns the next job to be run
func nextJob()*Job{
	return queue[0]
}

// remove a job from the list
func rmJob(job *Job, inList []*Job) (outList []*Job){
	// find index
	i:=0
	for;i<len(inList);i++{
		if inList[i] == job{break}
	}
	outList = append(inList[:i], inList[i+1:]...)
	return
}


func fillNodes(){

}


// returns the first free node + device
func freeDevice() (node *Node, device int){
	for _,n := range nodes{
		for d, busy := range n.devBusy{
			if !busy{
				node = n
				device = d
				return
			}
		}
	}
	return // nil,0 : nothing free
}
