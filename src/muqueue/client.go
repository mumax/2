//  This file is part of muQueue, a job scheduler for MuMax.
//  Copyright 2012  Arne Vansteenkiste.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.

package main

// Client main loop

import (
	"rpc"
	"fmt"
	"os"
)

func clientMain(args []string) {
	client, err := rpc.DialHTTP("tcp", "localhost"+PORT)
	check(err)
	var resp string
	user := os.Getenv("USER")
	args = append([]string{user}, args...)
	err2 := client.Call("RPC.Call", args, &resp)
	check(err2)
	fmt.Println(resp)
}
