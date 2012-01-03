//  This file is part of muQueue, a job scheduler for MuMax.
//  Copyright 2012  Arne Vansteenkiste.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.

package main

// Main program

import (
	"flag"
)

var flagServer *bool = flag.Bool("server", false, "Server mode")
var flagHost *string = flag.String("host", "localhost", "Server address")
var flagPort *string = flag.String("port", ":2527", "Network port")

func main() {
	flag.Parse()
	if *flagServer {
		serverMain()
	} else {
		clientMain()
	}
}
