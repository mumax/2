//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine


// This file implements RPC (Remote Procedure Call) from a client to an engine.
// Trying to call an undefined client method via its interpreter will automatically
// forward this call to the engine. In this way, both the client and engine methods
// are available to the scripting child process in a transparent way.
//
// The engine can run locally or connected over the network (see program flags).
//
// Author: Arne Vansteenkiste

import (
	. "mumax/common"
	"os"
	"io"
	"net"
	"rpc"
	"fmt"
)


// INTERNAL but exported because package rpc requires so.
// wraps the arguments for an rpc call to engine.RelfectCall
type CallArgs struct {
	Func string
	Args []string
}


// wraps an engine to allow its methods to be called over rpc
// via ReflectCall(funcname, args). This avoids having to write
// all of engine's methods in the special format required by rpc.
type engineRPCWrapper struct { // todo: rename
	ipc interpreter
}


func newEngineRPCWrapper(eng *Engine) *engineRPCWrapper {
	e := new(engineRPCWrapper)
	e.ipc.init(eng, nil)
	return e
}


// INTERNAL but exported because package rpc requires so.
// this rpc-exported method uses an interpreter to parse the function name and argument values
// (strings) in the CallArgs argument, and calls the function using reflection. 
func (e *engineRPCWrapper) ReflectCall(args_ *CallArgs, reply *interface{}) os.Error {
	// TODO: error handling
	args := *args_
	ret := e.ipc.call(args.Func, args.Args)
	switch len(ret) {
	default:
		panic(Bug(fmt.Sprint("Too many return values for", args.Func)))
	case 0:
		*reply = nil // no return values
	case 1:
		*reply = ret[0]
	}
	return nil
}


// global simulation engine
var eng *Engine

// global engine wrapper for rpc
var engRPCWrap *engineRPCWrapper


// initializes an engine and starts listening for gob rpc calls on the port determined by flag_port
func listen() {
	initEngineWrapper()

	addr, err1 := net.ResolveTCPAddr("tcp", "localhost"+*flag_port)
	CheckErr(err1, ERR_IO)
	Debug("listen addr", addr)

	listener, err2 := net.ListenTCP("tcp", addr)
	CheckErr(err2, ERR_IO)
	Debug("listening...")

	conn, err3 := listener.Accept()
	CheckErr(err3, ERR_IO)
	Debug("connected", conn)
	rpc.ServeConn(conn)
	Debug("done serving")
}


// like listen, but used when the engine runs locally. A software pipe is used
// for the gob communication, avoiding actual network overhead.
func localConn() io.ReadWriteCloser {
	initEngineWrapper()

	Debug("running local engine")
	end1, end2 := net.Pipe()
	// TODO(a): this may be a race-condition (return connection before rpc actually runs)
	go func() {
		rpc.ServeConn(end1)
		Debug("engine: done serving")
	}()
	return end2
}


func initEngineWrapper() {
	Assert(engRPCWrap == nil)
	Assert(eng == nil)
	eng = newEngine()
	engRPCWrap = newEngineRPCWrapper(eng)
	//Debug("rpc.Register", engRPCWrap)
	rpc.RegisterName("engine", engRPCWrap)
}
