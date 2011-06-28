//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine


//
// Author: Arne Vansteenkiste

import (
		. "mumax/common"
		"net"
)

func serverMain(){

	listener, err2 := net.Listen(*flag_net, "localhost:" + *flag_port)
	CheckErr(err2, ERR_IO)
	Debug("listening...")

	conn, err3 := listener.Accept()
	CheckErr(err3, ERR_IO)
	Debug("connected", conn)

	eng := NewEngine()
	var server Server
	server.Init(eng, conn)
	server.Run()
}
