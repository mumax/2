//  This file is part of muQueue, a job scheduler for MuMax.
//  Copyright 2012  Arne Vansteenkiste.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.

package main

import(
"fmt"
"exec"
)

func dispatch(job *Job, node *Node, dev int) string{
	log("dispatch", job, node, dev)
	
	cmd := exec.Command("mumax2", "-s", "-gpu="+fmt.Sprint(dev), job.file)
	go func(){
		cmd.Run()
	}()	
	return fmt.Sprint("dispatched ", job, " to ", node.hostname, ":", dev)
}


func dispatchNext() string{
	node, dev := freeDevice()
	if node == nil{
		return "No free device"
	}
	return dispatch(nextJob(), node, dev)
}

func init() {
	api["dispatch"] = dispatchManual
}

// Manual dispatch
func dispatchManual(user *User, args []string) string {
	if len(args) == 0 {
		return dispatchNext()
	}

	resp := ""
	return resp
}



