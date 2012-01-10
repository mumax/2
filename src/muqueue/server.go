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
	"runtime/debug"
	"strings"
)

const PORT = ":2527"

var do_recover bool = false // server should recover from errors?

func serverMain() {
	go runSched() // start scheduler loop

	AddUser("root", "-", 0) // add super user

	if len(os.Args) >= 3 {
		runFile(os.Args[2]) // read commands from config file first
	}

	do_recover = true // now that the config was read, recover errors

	runRPC() // loops forever
}

// run the rpc server
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

// passes the command to the scheduler, who will
// callback whenever he's ready.
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

// called by scheduler when he's ready to serve the command
// processes a command issued by user
func serveCommand(words []string) (response string) {
	defer func() {
		err := recover()
		if err != nil {
			msg := fmt.Sprint(err, "\n", string(debug.Stack()))
			response = msg
			if !do_recover {
				panic(err)
			}
		}
	}()
	log(words)
	username := words[0]
	command := words[1]
	args := words[2:]

	f, ok := api[command]
	if !ok {
		options := ""
		for k, _ := range api {
			options += " " + k
		}
		return "Not a valid command: " + command + "\nDid you mean one of these?\n" + options
	}
	return f(GetUser(username), args)
}

// reads the file and executes the commands
func runFile(file string) {
	log("reading", file)
	in, err := os.Open(file)
	check(err)
	for line, eof := ReadLine(in); eof == false; line, eof = ReadLine(in) {
		if strings.HasPrefix(line, "#") {
			continue
		}
		words := strings.Split(line, " ")
		ret := serveCommand(append([]string{"root"}, words...))
		fmt.Println(ret)
	}
}
