//  This file is part of muQueue, a job scheduler for MuMax.
//  Copyright 2012  Arne Vansteenkiste.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.

package main

// Network front-end

import (
	"net"
)

// initialize the network front-end
func initNet() {

}

// run the network front-end
func runNet() {
	const NET = "tcp"

	addr, err1 := net.ResolveTCPAddr(NET, *flagHost+*flagPort)
	check(err1)

	listener, err2 := net.ListenTCP(NET, addr)
	check(err2)
	log("listening on", addr)

	for {
		conn, err3 := listener.AcceptTCP()
		check(err3)
		log("connected to", conn.RemoteAddr())
		serveConn(conn)
	}
}

func serveConn(conn net.Conn) {
	line, _ := readLine(conn)
	//log("read", line)

	resp := make(chan string)
	input <- &Cmd{line, resp}

	conn.Write([]byte(<-resp))
	conn.Close()
}
