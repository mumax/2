//  This file is part of muQueue, a job scheduler for MuMax.
//  Copyright 2012  Arne Vansteenkiste.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.

package main

// Client main loop

import (
	"fmt"
	. "mumax/common"
	"os"
	"path"
	"rpc"
	"strings"
)

func clientMain(args []string) {
	client, err1 := rpc.DialHTTP("tcp", "localhost"+PORT)
	check(err1)

	// Hack for "add" command:
	// resolve file to full path and check if it exists
	if args[0] == "add" {
		shortFile := args[len(args)-1]
		file := ReadLink(shortFile)
		args[len(args)-1] = file
		if !FileExists(file) {
			err("file", file, "does not exist")
		}
	}

	var resp string
	user := os.Getenv("USER")
	args = append([]string{user}, args...)
	err2 := client.Call("RPC.Call", args, &resp)
	check(err2)
	fmt.Println(strings.Trim(resp, "\n"))
}

// return full path of file
func ReadLink(file string) (fullpath string) {
	if path.IsAbs(file) {
		return file
	}
	wd, err := os.Getwd()
	if err == nil {
		return path.Join(wd, file)
	} else {
		log(err)
	}
	return file
}
