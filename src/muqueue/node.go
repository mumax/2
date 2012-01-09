//  This file is part of muQueue, a job scheduler for MuMax.
//  Copyright 2012  Arne Vansteenkiste.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.

package main

// Worker node

import (
	"fmt"
)

// Worker node
type Node struct {
	id       int
	hostname string
	loginCmd []string
	devBusy  []bool // GPU[i] in use?
	group    string // group that owns this node
	// draining bool // stop using this node	
}

func NewNode(hostname string, NDev int, group string, loginCmd []string) *Node {
	n := new(Node)
	n.hostname = hostname
	n.group = group
	n.loginCmd = loginCmd
	n.devBusy = make([]bool, NDev)
	lastNodeId++
	n.id = lastNodeId
	return n
}

var lastNodeId int

func (n *Node) String() string {
	return fmt.Sprint("node", n.id, "(", n.hostname, ",", n.NDevice(), "GPU", " "+n.group, ")")
}

func (n *Node) NDevice() int {
	return len(n.devBusy)
}

// returns if the node is completely busy
// (not a single device free)
func (n *Node) Busy() bool {
	for _, busy := range n.devBusy {
		if !busy {
			return false
		}
	}
	return true
}
