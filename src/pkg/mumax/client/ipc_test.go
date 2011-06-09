//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package client

import (
	"testing"
	//"os"
)

func TestIPC(test *testing.T) {
	// do not crash on panic, but fail
	defer func() {
		if err := recover(); err != nil {
			test.Fatal(err)
		}
	}()

	recv := &St{1}
	var c interpreter
	c.init(recv)
	retval := c.call("Get", []string{})[0]
	if retval.(int) != 1 {
		test.Fatal("Expected", 1, "got", retval)
	}
}

type St struct {
	It int
}

func (s *St) Get() int {
	return s.It
}

func (s *St) private() int {
	return 3
}
