//  This file is part of muQueue, a job scheduler for MuMax.
//  Copyright 2012  Arne Vansteenkiste.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.

package main

// Client main loop

import (
	"net"
	"flag"
	"io/ioutil"
)

func clientMain() {
	if flag.NArg() == 0 {
		err("need command line argument")
	}

	text := flag.Arg(0)
	conn := dialServer()
	check(conn.SetWriteTimeout(1))
	_, err1 := conn.Write([]byte(text))
	check(err1)

	resp, err2 := ioutil.ReadAll(conn)
	check(err2)
	log(string(resp))
}

// connects to the job server
func dialServer() net.Conn {
	const NET = "tcp"

	raddr, err1 := net.ResolveTCPAddr(NET, *flagHost+*flagPort)
	check(err1)

	laddr, err2 := net.ResolveTCPAddr(NET, "localhost:0")
	check(err2)

	conn, err3 := net.DialTCP(NET, laddr, raddr)
	check(err3)
	log("connected to", conn.RemoteAddr())
	return conn
}
