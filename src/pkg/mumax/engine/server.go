//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine


// This file implements the mumax server. 
// The server exposes an engine via RPC (Remote Procedure Call).
// The server can run locally or connected over the network (see program flags).
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


// Wraps an engine to allow its methods to be called over rpc
// via ReflectCall(funcname, args). This avoids having to write
// all of engine's methods in the special format required by rpc.
type Server struct {
	conn      io.ReadWriteCloser
	ipc       Interpreter
	rpcServer *rpc.Server

	eng *Engine
}


func (s *Server) Init(eng *Engine, conn io.ReadWriteCloser) {
	s.conn = conn
	s.ipc.Init(eng, nil)
	s.eng = eng
	s.rpcServer = rpc.NewServer()
	s.rpcServer.RegisterName("server", s)
}


func (s *Server) Run() {
	s.rpcServer.ServeConn(s.conn)
}


func (s *Server) ServeConn(conn io.ReadWriteCloser) {
	rpc.ServeConn(conn)
	Debug("done serving")
}

// ---------------


// INTERNAL but exported because package rpc requires so.
// this rpc-exported method uses an interpreter to parse the function name and argument values
// (strings) in the ReflectCallArgs argument, and calls the function using reflection. 
func (e *Server) ReflectCall(args_ *ReflectCallArgs, reply *interface{}) os.Error {
	// TODO: error handling
	args := *args_
	ret := e.ipc.Call(args.Func, args.Args)
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


// INTERNAL but exported because package rpc requires so.
// wraps the arguments for an rpc call to engine.RelfectCall
type ReflectCallArgs struct {
	Func string
	Args []string
}


// initializes an engine and starts listening for gob rpc calls on the port determined by flag_port
//func serverMain() {
//	initServer()
//
//	addr, err1 := net.ResolveTCPAddr("tcp", "localhost:"+*flag_port)
//	CheckErr(err1, ERR_IO)
//	Debug("listen addr", addr)
//
//	listener, err2 := net.ListenTCP("tcp", addr)
//	CheckErr(err2, ERR_IO)
//	Debug("listening...")
//
//	conn, err3 := listener.Accept()
//	CheckErr(err3, ERR_IO)
//	Debug("connected", conn)
//	s.ServeConn(conn)
//}
//
//
//// like listen, but used when the engine runs locally. A software pipe is used
//// for the gob communication, avoiding actual network overhead.
//func localConn() io.ReadWriteCloser {
//	initServer()
//
//	Debug("running local engine")
//	end1, end2 := net.Pipe()
//	// TODO(a): this may be a race-condition (return connection before rpc actually runs)
//	go func() {
//		rpc.ServeConn(end1)
//		Debug("engine: done serving")
//	}()
//	return end2
//}
