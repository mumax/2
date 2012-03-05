//  This file is part of muQueue, a job scheduler for MuMax.
//  Copyright 2012  Arne Vansteenkiste.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.

package main

// Implementation of "dump" command

import (
	"bytes"
	"fmt"
	"io"
	. "mumax/common"
)

func init() {
	api["dump"] = dump
}

// dumps all job so they can be re-loaded later
func dump(user *User, args []string) string {
	var out io.Writer
	if len(args) == 0 {
		out = bytes.NewBuffer(make([]byte, 4096))
	} else {
		f := OpenWRONLY(args[0])
		defer f.Close()
		out = f
	}

	fmt.Fprintln(out, "#", len(queue), "jobs")
	for _, job := range queue {
		if job.status == QUEUED {
			command := ""
			for _, c := range job.command {
				command += " " + c
			}
			fmt.Fprint(out, "add -user=", job.user.name, " -gpus=", job.ndev, " -nice=", job.nice, command, "\n")
		}
	}

	if len(args) == 0 {
		return string(out.(*bytes.Buffer).Bytes())
	} //else{
	return "wrote dump to " + args[0] + " (on server)"
}
