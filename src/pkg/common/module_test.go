//  This file is part of MuMax, a high-perfomrance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package common

// Author: Arne Vansteenkiste

import (
	"testing"
)


func TestModule(test *testing.T) {
	m := ModuleLoad(GetExecDir() + "testmodule.ptx")
	c := m.MakeClosure("testMemset", 3)

	dev := NewArray(1, []int{4, 4, 4})

	for i := range c.DeviceClosure {
		c.DeviceClosure[i].SetArg(0, float32(42))
		c.DeviceClosure[i].SetArg(1, dev.DevicePtr(i))
		c.DeviceClosure[i].SetArg(2, dev.splice.list.slice[i].Len())
	}

}
