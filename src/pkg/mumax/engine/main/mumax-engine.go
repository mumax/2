//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package main

import (
	. "mumax/common"
	"mumax/engine"
	"flag"
	"runtime"
	"os"
	"fmt"
)

const WELCOME = `MuMax 2.0.0.70 FD Multiphysics Engine (C) Arne Vansteenkiste & Ben Van de Wiele, Ghent University.`

var (
	logFile *string = flag.String("log", "mumax.log", "Specify the log file.")
	rpcType *string = flag.String("rpc", "shell", "Specify RPC protocol: shell, (go, json)")
	interactive *bool = flag.Bool("i", true, "interactive mode")
)

func main() {
	initialize()
	defer cleanup()

	var eng engine.Engine
	rpc := chooseRPC(*rpcType, &eng)

	if *interactive{
		runInteractive(rpc)
	}else{
		rpc.ReadFrom(os.Stdin)
	}
}


func initialize(){
	flag.Parse()
	InitLogger(*logFile)
	Println(WELCOME)
	Debug("Go version:", runtime.Version())
}

const(
	UNKNOWN_RPC = "unknown RPC: -rpc=%v"
)

func chooseRPC(rpcType string, eng *engine.Engine) engine.RPC{
	switch rpcType{
	default: panic(InputErr(fmt.Sprintf(UNKNOWN_RPC, rpcType)))
	case "shell": return engine.NewRefshRPC(eng)
	}
	panic(Bug("Bug"))
	return nil
}

const PROMPT = "mumax> "

func runInteractive(rpc engine.RPC){
	var err interface{}	= "dummy"
	for err != nil{	
		func(){
			defer func(){
				err = recover()
				if err != nil {fmt.Fprintln(os.Stderr, err)}
			}()
		rpc.ReadFrom(os.Stdin)
		}()
	}
}

func cleanup(){
	Println("Finished.")
}
