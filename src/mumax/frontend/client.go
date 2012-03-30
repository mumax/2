//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package frontend

// The client implements Inter-Process-Communication
// between mumax and a scripting language.
// Author: Arne Vansteenkiste

import (
	//"fmt"
	. "mumax/common"
	"mumax/engine"
	"net"
	"bufio"
	"time"
)

type Client struct {
	outputDir string
	ipc                  jsonRPC
	api                  engine.API
	wire			 	 net.Conn
	logWait              chan int // channel to wait for completion of go logStream()
}

// Initializes the mumax client to parse infile, write output
// to outdir and connect to a server over conn.
func (c *Client) Init(outputDir string) {
	c.outputDir = outputDir
	
	// CheckErr(os.Setenv("MUMAX2_OUTPUTDIR", c.outputDir), ERR_IO)
	engine.Init()
	engine.GetEngine().SetOutputDirectory(outputDir)
	c.api = engine.API{engine.GetEngine()}
}

// Start interpreter sub-command and communicate over fifos in the output dir.
func (c *Client) Run() {

	s_infifo := bufio.NewReader(c.wire)
	s_outflush := bufio.NewWriter(c.wire) 	
	s_outfifo := s_outflush
	
	c.ipc.Init(s_infifo, s_outfifo, *s_outflush, c.api)
	c.ipc.Run()
	Debug("Client is now disconnected")	
	
}


// returns a channel that will signal when the file has appeared
func pollFile(fname string) (waiter chan (int)) {
	waiter = make(chan (int))
	go func() {
		for !FileExists(fname) {
			time.Sleep(10)
		}
		waiter <- 1
	}()
	return
}
