//  This file is part of muQueue, a job scheduler for MuMax.
//  Copyright 2012  Arne Vansteenkiste.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.

package main

// Server main loop

import (
	"rpc"
	"net"
	"http"
	"fmt"
	"os"
)

const PORT = ":2527"

func serverMain() {
	go runSched() // start scheduler loop
	runRPC()      // loops forever
}

func runRPC() {
	RPC := new(RPC)
	rpc.Register(RPC)
	rpc.HandleHTTP()
	l, e := net.Listen("tcp", PORT)
	if e != nil {
		err("listen error:", e)
	}
	log("Listening on port " + PORT)
	http.Serve(l, nil)
}

type RPC int // dummy type

func (r *RPC) Call(args []string, resp *string) (err os.Error) {
	defer func() {
		e := recover()
		if e != nil {
			err = os.NewError(fmt.Sprint(e))
		}
	}()
	respChan := make(chan string)
	input <- &Cmd{args, respChan}
	*resp = <-respChan
	return
}
