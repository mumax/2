//  This file is part of muQueue, a job scheduler for MuMax.
//  Copyright 2012  Arne Vansteenkiste.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.

package main

// Command line arg parsing

import (
	"strings"
)

func parse(args []string, knownflags ...string) (files []string, flags map[string]string) {
	flags = make(map[string]string)
	flagsdone := false
	for _, arg := range args {
		if !flagsdone && strings.HasPrefix(arg, "-") {
			split := strings.Split(arg, "=")
			key := strings.Trim(split[0], "-")
			value := split[1]
			flags[key] = value
		} else {
			files = append(files, arg)
			flagsdone = true
		}
	}

	return
}
