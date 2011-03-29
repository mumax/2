//  This file is part of MuMax, a high-perfomrance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

// Author: Arne Vansteenkiste

package common

import (
	"testing"
	"fmt"
	"cuda"
)


func init() {
	UseDebugDevices()
}


// Test if getDevices() works
func TestDevices(t *testing.T) {
	devices := getDevices()
	for i, d := range devices {
		fmt.Println("Device ", i)
		fmt.Println(cuda.GetDeviceProperties(d))
		fmt.Println()
		fmt.Println()
	}
}

// Test splice alloc/free
func TestSpliceAlloc(t *testing.T) {
	for i := 0; i < 10; i++ {
		s := NewSplice(256 * 1024 * 1024)
		s.Free()
	}
}


func TestSpliceCopy(t *testing.T){

}
