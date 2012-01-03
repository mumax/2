//  This file is part of muQueue, a job scheduler for MuMax.
//  Copyright 2012  Arne Vansteenkiste.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.

package main

// Implementation of "remote" command

import (
	"fmt"
)

func init() {
	api["remote"] = remote
}

func remote(user *User, args []string) string {
	if len(args) == 0 {
		return remoteShow()
	}
	subcommand := args[0]
	switch subcommand{
		default: return "Unknown subcommand: " + subcommand
		case "show": return remoteShow()
		//case "add":return remoteAdd(args[1:])
	}
	return"<internal error>"
}


// Show remote worker nodes
func remoteShow()string{
	info:=""
	for _,n:=range nodes{
		info += fmt.Sprintln(n)
	}
	return info
}

//func remoteAdd(args[]string)string{
	//nodes=append(nodes, 
//}
