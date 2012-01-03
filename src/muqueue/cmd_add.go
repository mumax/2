//  This file is part of muQueue, a job scheduler for MuMax.
//  Copyright 2012  Arne Vansteenkiste.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.

package main

// Implementation of "add" command

import ("fmt")

func init(){
	api["add"]=add
}


// adds a job
func add(user *User, args []string) string {
	if len(args) == 0 {
		return "Nothing specified, nothing added.\nMaybe you wanted to say 'add file'?"
	}
	resp := ""
	for _, file := range args {
		queue = append(queue, NewJob(user, file))
		resp += fmt.Sprint("add ", file, " [#", len(queue), "]\n")
	}
	return resp
}
