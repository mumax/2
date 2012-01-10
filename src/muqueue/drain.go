//  This file is part of muQueue, a job scheduler for MuMax.
//  Copyright 2012  Arne Vansteenkiste.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.

package main

// Implementation of "drain and fill" command


import ()

func init() {
	api["drain"] = drain
	api["fill"] = fill
}

// drains a node: do not start new jobs on it
func drain(user *User, args []string) string {
	node, resp := resolveNode(args)
	if node == nil {
		return resp
	}
	node.drain = true
	return "Draining node " + args[0] + ". Use 'fill " + args[0] + "' to start using it again."
}

// fill a node: start running jobs on it
func fill(user *User, args []string) string {
	node, resp := resolveNode(args)
	if node == nil {
		return resp
	}
	node.drain = false
	fillNodes()
	return "now using node " + args[0]
}

// get a node from host name (arg[0])
func resolveNode(args []string) (n *Node, resp string) {
	if len(args) == 0 {
		resp = "need an argument (hostname)"
		return
	}
	hostname := args[0]
	_, ok := nodemap[hostname]
	if !ok {
		resp = "no such node: " + hostname
		return
	}
	n = nodemap[hostname]
	return
}
