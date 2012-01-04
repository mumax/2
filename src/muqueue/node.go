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
	hostname string
	devBusy  []bool // GPU[i] in use?
}

func NewNode(hostname string, NDev int) *Node {
	n := new(Node)
	n.hostname = hostname
	n.devBusy = make([]bool, NDev)
	return n
}

func (n *Node) String() string {
	return fmt.Sprint(n.hostname)
}

func (n *Node) NDevice() int {
	return len(n.devBusy)
}
