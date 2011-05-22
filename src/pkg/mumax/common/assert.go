//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package common

// This file implements assertions.
// Author: Arne Vansteenkiste

import (
	"fmt"
)


const MSG_ASSERTIONFAILED = "Assertion failed."


// Panics if test is false
func Assert(test bool) {
	if !test {
		panic(Bug(MSG_ASSERTIONFAILED))
	}
}


// Panics if test is false, printing the message.
func AssertMsg(test bool, msg ...interface{}) {
	if !test {
		panic(Bug(fmt.Sprint(msg...)))
	}
}


// Panics if the slice are not equal.
// Used to check for equal tensor sizes.
func AssertEqual(a, b []int) {
	if len(a) != len(b) {
		panic(Bug(MSG_ASSERTIONFAILED))
	}
	for i, a_i := range a {
		if a_i != b[i] {
			panic(Bug(MSG_ASSERTIONFAILED))
		}
	}
}
