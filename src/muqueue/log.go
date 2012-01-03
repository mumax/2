//  This file is part of muQueue, a job scheduler for MuMax.
//  Copyright 2012  Arne Vansteenkiste.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.

package main

// Log and error handling

import (
	"fmt"
	"os"
)

// Exits with optional message if error is non-nil
func check(err os.Error, msg ...interface{}) {
	if err != nil {
		fmt.Fprint(os.Stderr, msg...)
		fmt.Fprintln(os.Stderr, " ", err)
		os.Exit(1)
	}
}

func log(msg ...interface{}) {
	fmt.Println(msg...)
}

func err(msg ...interface{}){
		fmt.Fprintln(os.Stderr, msg...)
		os.Exit(1)
}
