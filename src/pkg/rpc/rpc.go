//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// This file implements a 2-way RPC (Remote Procedure Call) protocol
// between an engine Server and Client. It is similar to Go's rpc package
// but allows the Server to call the Client and vice-versa over a single
// connection.  Hence, there is no distinction between an rpc server and
// client. Both are represented by engine.RPC.
// The usage is somewhat simpeler than Go's rpc package:
// ...
//
// TODO(a): Furthermore, this RPC is tweaked to handle large float32 arrays efficiently.
//
// Author: Arne Vansteenkiste

import (
	"io"
	"gob"
	"os"
	"fmt"
)

// INTERNAL
type RPC struct {
	conn    io.ReadWriteCloser // Connection with the other side
	encoder *gob.Encoder
	decoder *gob.Decoder
	obj     interface{} // Receiver object whose methods are exported
	call    Call        // 
	resp    Response
}

func (rpc *RPC) Init() {

}

func NewRPC() *RPC {
	rpc := new(RPC)
	rpc.Init()
	return rpc
}

// Register the receiver object whose methods are exported.
func (rpc *RPC) Register(obj interface{}) {
	rpc.obj = obj
}

func (rpc *RPC) ServeConn(conn io.ReadWriteCloser) {
	rpc.conn = conn
	rpc.encoder = gob.NewEncoder(conn)
	rpc.decoder = gob.NewDecoder(conn)

	var err os.Error
	for err == nil {
		err = rpc.decoder.Decode(&rpc.call)
		if err == nil {
			fmt.Println(rpc.call.Method, rpc.call.Args)
			err = rpc.encoder.Encode(&rpc.resp)
		}
	}
}


func (rpc *RPC) Call(method string, args []interface{}) []interface{} {
	// here be dragons

	println("TODO")
	return []interface{}{}
}

// INTERNAL
type Call struct {
	Method string
	Args   []interface{}
	//Bigdata []float32
}

// INTERNAL
type Response struct {
	Value []interface{}
	Error interface{}
	//Bigdata []float32
}
