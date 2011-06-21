//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine


import (
	. "mumax/common"
	"os"
	"io"
	"net"
	"rpc"
)


type CallArgs struct {
	Func string
	Args []string
}


type Export struct {

}


func (e *Export) ReflectCall(args *CallArgs, reply *interface{}) os.Error {
	return nil
}


var export *Export

func listen() {
	Assert(export == nil)
	export = new(Export)
	Debug("rpc.Register", export)
	rpc.RegisterName("engine", export)

	addr, err1 := net.ResolveTCPAddr("tcp", "localhost"+*Flag_port)
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


func LocalConn() io.ReadWriteCloser {
	Assert(export == nil)
	export = new(Export)
	Debug("rpc.Register", export)
	rpc.RegisterName("engine", export)

	Debug("running local engine")
	end1, end2 := net.Pipe()
	go func() {
		rpc.ServeConn(end1)
		Debug("engine: done serving")
	}()
	return end2
}
