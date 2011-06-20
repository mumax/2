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
	"rpc"
)


type CallArgs struct {
	Func string
	Args []interface{}
}


type Export struct {

}


func (e *Export) Call(args *CallArgs, reply *interface{}) os.Error {

	return nil
}


var export *Export

func Listen() {
	Assert(export == nil)
	export = new(Export)
	rpc.RegisterName("engine", export)
	//rpc.ServeConn()
}
